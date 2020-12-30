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

#include "legion/legion_ops.h"
#include "legion/legion_trace.h"
#include "legion/legion_views.h"
#include "legion/legion_context.h"
#include "legion/legion_replication.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

#ifdef DEBUG_LEGION_COLLECTIVES
    /////////////////////////////////////////////////////////////
    // Collective Check Reduction
    /////////////////////////////////////////////////////////////
    
    /*static*/ const long CollectiveCheckReduction::IDENTITY = -1;
    /*static*/ const long CollectiveCheckReduction::identity = IDENTITY;
    /*static*/ const long CollectiveCheckReduction::BAD = -2;
    /*static*/ const ReductionOpID CollectiveCheckReduction::REDOP = 
                                                MAX_APPLICATION_REDUCTION_ID;

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CollectiveCheckReduction::apply<true>(LHS &lhs, RHS rhs)
    //--------------------------------------------------------------------------
    {
      assert(rhs > IDENTITY);
      if (lhs != IDENTITY)
      {
        if (lhs != rhs)
          lhs = BAD;
      }
      else
        lhs = rhs;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CollectiveCheckReduction::apply<false>(LHS &lhs, RHS rhs)
    //--------------------------------------------------------------------------
    {
      volatile LHS *ptr = &lhs;
      LHS temp = *ptr;
      while ((temp != BAD) && (temp != rhs))
      {
        if (temp != IDENTITY)
          temp = __sync_val_compare_and_swap(ptr, temp, BAD);
        else
          temp = __sync_val_compare_and_swap(ptr, temp, rhs); 
      }
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CollectiveCheckReduction::fold<true>(RHS &rhs1, RHS rhs2)
    //--------------------------------------------------------------------------
    {
      assert(rhs2 > IDENTITY);
      if (rhs1 != IDENTITY)
      {
        if (rhs1 != rhs2)
          rhs1 = BAD;
      }
      else
        rhs1 = rhs2;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CollectiveCheckReduction::fold<false>(RHS &rhs1, RHS rhs2)
    //--------------------------------------------------------------------------
    {
      volatile RHS *ptr = &rhs1;
      RHS temp = *ptr;
      while ((temp != BAD) && (temp != rhs2))
      {
        if (temp != IDENTITY)
          temp = __sync_val_compare_and_swap(ptr, temp, BAD);
        else
          temp = __sync_val_compare_and_swap(ptr, temp, rhs2);
      }
    }

    /////////////////////////////////////////////////////////////
    // Check Reduction
    /////////////////////////////////////////////////////////////
    
    /*static*/ const CloseCheckReduction::CloseCheckValue 
      CloseCheckReduction::IDENTITY = CloseCheckReduction::CloseCheckValue();
    /*static*/ const CloseCheckReduction::CloseCheckValue
      CloseCheckReduction::identity = IDENTITY;
    /*static*/ const ReductionOpID CloseCheckReduction::REDOP = 
                                              MAX_APPLICATION_REDUCTION_ID + 1;

    //--------------------------------------------------------------------------
    CloseCheckReduction::CloseCheckValue::CloseCheckValue(void)
      : operation_index(0), region_requirement_index(0),
        barrier(RtBarrier::NO_RT_BARRIER), region(LogicalRegion::NO_REGION), 
        partition(LogicalPartition::NO_PART), is_region(true), read_only(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CloseCheckReduction::CloseCheckValue::CloseCheckValue(
        const LogicalUser &user, RtBarrier bar, RegionTreeNode *node, bool read)
      : operation_index(user.op->get_ctx_index()), 
        region_requirement_index(user.idx), barrier(bar),
        is_region(node->is_region()), read_only(read)
    //--------------------------------------------------------------------------
    {
      if (is_region)
        region = node->as_region_node()->handle;
      else
        partition = node->as_partition_node()->handle;
    }

    //--------------------------------------------------------------------------
    bool CloseCheckReduction::CloseCheckValue::operator==(const
                                                     CloseCheckValue &rhs) const
    //--------------------------------------------------------------------------
    {
      if (operation_index != rhs.operation_index)
        return false;
      if (region_requirement_index != rhs.region_requirement_index)
        return false;
      if (barrier != rhs.barrier)
        return false;
      if (read_only != rhs.read_only)
        return false;
      if (is_region != rhs.is_region)
        return false;
      if (is_region)
      {
        if (region != rhs.region)
          return false;
      }
      else
      {
        if (partition != rhs.partition)
          return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CloseCheckReduction::apply<true>(LHS &lhs, RHS rhs)
    //--------------------------------------------------------------------------
    {
      // Only copy over if LHS is the identity
      // This will effectively do a broadcast of one value
      if (lhs == IDENTITY)
        lhs = rhs;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CloseCheckReduction::apply<false>(LHS &lhs, RHS rhs)
    //--------------------------------------------------------------------------
    {
      // Not supported at the moment
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CloseCheckReduction::fold<true>(RHS &rhs1, RHS rhs2)
    //--------------------------------------------------------------------------
    {
      // Only copy over if RHS1 is the identity
      // This will effectively do a broadcast of one value
      if (rhs1 == IDENTITY)
        rhs1 = rhs2;
    }

    //--------------------------------------------------------------------------
    template<>
    /*static*/ void CloseCheckReduction::fold<false>(RHS &rhs1, RHS rhs2)
    //--------------------------------------------------------------------------
    {
      // Not supported at the moment
      assert(false);
    }
#endif // DEBUG_LEGION_COLLECTIVES

    /////////////////////////////////////////////////////////////
    // Repl Individual Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndividualTask::ReplIndividualTask(Runtime *rt)
      : IndividualTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplIndividualTask::ReplIndividualTask(const ReplIndividualTask &rhs)
      : IndividualTask(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplIndividualTask::~ReplIndividualTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplIndividualTask& ReplIndividualTask::operator=(
                                                  const ReplIndividualTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_individual_task();
      owner_shard = 0;
      sharding_functor = UINT_MAX;
      sharding_function = NULL;
      mapped_collective_id = UINT_MAX;
      future_collective_id = UINT_MAX;
      mapped_collective = NULL;
      future_collective = NULL;
#ifdef DEBUG_LEGION
      sharding_collective = NULL; 
#endif
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      if (mapped_collective != NULL)
        delete mapped_collective;
      if (future_collective != NULL)
        delete future_collective;
      deactivate_individual_task();
      runtime->free_repl_individual_task(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // We might be able to skip this if the sharding function was already
      // picked for us which occurs when we're part of a must-epoch launch
      if (sharding_function == NULL)
      {
        // Do the mapper call to get the sharding function to use
        if (mapper == NULL)
          mapper = runtime->find_mapper(current_proc, map_id); 
        Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
        Mapper::SelectShardingFunctorOutput output;
        output.chosen_functor = UINT_MAX;
        mapper->invoke_task_select_sharding_functor(this, input, &output);
        if (output.chosen_functor == UINT_MAX)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Mapper %s failed to pick a valid sharding functor for "
                        "task %s (UID %lld)", mapper->get_mapper_name(),
                        get_task_name(), get_unique_id())
        this->sharding_functor = output.chosen_functor;
        sharding_function = 
          repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      }
#ifdef DEBUG_LEGION
      assert(sharding_function != NULL);
      // In debug mode we check to make sure that all the mappers
      // picked the same sharding function
      assert(sharding_collective != NULL);
      // Contribute the result
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() && 
          !sharding_collective->validate(this->sharding_functor))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s chose different sharding functions "
                      "for individual task %s (UID %lld) in %s "
                      "(UID %lld)", mapper->get_mapper_name(), get_task_name(), 
                      get_unique_id(), parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id())
#endif 
      // Now we can do the normal prepipeline stage
      IndividualTask::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sharding_function != NULL);
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Figure out whether this shard owns this point
      if (sharding_space.exists())
      {
        Domain shard_domain;
        runtime->forest->find_launch_space_domain(sharding_space, shard_domain);
        owner_shard = sharding_function->find_owner(index_point, shard_domain);
      }
      else
        owner_shard = sharding_function->find_owner(index_point, index_domain);
      // If we're recording then record the owner shard
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert(!is_remote());
        assert((tpl != NULL) && tpl->is_recording());
#endif
        tpl->record_owner_shard(trace_local_id, owner_shard);
      }
      if (runtime->legion_spy_enabled)
        LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
#ifdef DEBUG_LEGION
      assert(mapped_collective == NULL);
#endif
      mapped_collective = 
        new ShardEventTree(repl_ctx, owner_shard, mapped_collective_id);
      // If we own it we go on the queue, otherwise we complete early
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
#ifdef LEGION_SPY
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We don't own it, so we can pretend like we
        // mapped and executed this copy already
        // Before we do this though we have to get the version state
        // names for any writes so we can update our local state
        const RtEvent local_done = mapped_collective->get_local_event();
        shard_off(local_done);
      }
      else // We own it, so it goes on the ready queue
      {
        // Signal the tree when we are done our mapping
        mapped_collective->signal_tree(mapped_event);
        // Then we can do the normal analysis
        IndividualTask::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      // Figure out if we're the one to do the replay
#ifdef DEBUG_LEGION
      assert(!is_remote());
      assert(tpl != NULL);
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
      assert(sharding_collective != NULL);
      sharding_collective->elide_collective();
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      owner_shard = tpl->find_owner_shard(trace_local_id);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        if (runtime->legion_spy_enabled)
        {
          for (unsigned idx = 0; idx < regions.size(); idx++)
            TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
        }
#ifdef LEGION_SPY
        LegionSpy::log_replay_operation(unique_op_id);
#endif
        shard_off(RtEvent::NO_RT_EVENT);
        resolve_speculation();
      }
      else
        IndividualTask::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      if (launched)
        return;
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        sharding_collective->elide_collective();
#endif
      IndividualTask::resolve_false(speculated, launched);
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::shard_off(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      // Still need this to record that this operation is done for LegionSpy
      LegionSpy::log_operation_events(unique_op_id, 
          ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
      complete_mapping(mapped_precondition);
      if ((must_epoch == NULL) && 
          ((speculation_state != RESOLVE_FALSE_STATE) || false_guard.exists()))
      {
#ifdef DEBUG_LEGION
        ReplicateContext *repl_ctx = 
          dynamic_cast<ReplicateContext*>(parent_ctx);
        assert(repl_ctx != NULL);
        assert(future_collective == NULL);
#else
        ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
        future_collective = new FutureBroadcast(repl_ctx, 
                future_collective_id, owner_shard, result.impl);
        const RtEvent future_ready = 
          future_collective->perform_collective_wait(false/*block*/);
        // Do the stuff to record that this is mapped and executed
        complete_execution(future_ready);
      }
      else
        complete_execution();
      trigger_children_complete();
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Before doing the normal thing we have to exchange broadcast/receive
      // the future result, can skip this though if we're part of a must epoch
      // We should also skip this if we were predicated false
      if ((must_epoch == NULL) && 
          ((speculation_state != RESOLVE_FALSE_STATE) || false_guard.exists()) 
          && (owner_shard == repl_ctx->owner_shard->shard_id))
      {
#ifdef DEBUG_LEGION
        assert(future_collective == NULL);
#endif
        future_collective = new FutureBroadcast(repl_ctx, 
                future_collective_id, owner_shard, result.impl);
        future_collective->broadcast_future();
      }
      IndividualTask::trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
      mapped_collective_id = 
        ctx->get_next_collective_index(COLLECTIVE_LOC_0);
      future_collective_id = 
        ctx->get_next_collective_index(COLLECTIVE_LOC_1);
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::set_sharding_function(ShardingID functor,
                                                   ShardingFunction *function)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch != NULL);
      assert(sharding_function == NULL);
#endif
      sharding_functor = functor;
      sharding_function = function;
    }

    /////////////////////////////////////////////////////////////
    // Repl Index Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndexTask::ReplIndexTask(Runtime *rt)
      : IndexTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplIndexTask::ReplIndexTask(const ReplIndexTask &rhs)
      : IndexTask(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplIndexTask::~ReplIndexTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplIndexTask& ReplIndexTask::operator=(const ReplIndexTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_index_task();
      sharding_functor = UINT_MAX;
      sharding_function = NULL;
      reduction_collective = NULL;
      output_size_collective = NULL;
#ifdef DEBUG_LEGION
      sharding_collective = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_index_task();
      if (reduction_collective != NULL)
      {
        delete reduction_collective;
        reduction_collective = NULL;
      }
      if (output_size_collective != NULL)
      {
        delete output_size_collective;
        output_size_collective = NULL;
      }
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      unique_intra_space_deps.clear();
      runtime->free_repl_index_task(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // We might be able to skip this if the sharding function was already
      // picked for us which occurs when we're part of a must-epoch launch
      if (sharding_function == NULL)
        select_sharding_function(repl_ctx);
#ifdef DEBUG_LEGION
      assert(sharding_function != NULL);
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s chose different sharding functions "
                      "for index task %s (UID %lld) in %s (UID %lld)", 
                      mapper->get_mapper_name(), get_task_name(), 
                      get_unique_id(), parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id())
#endif
      // If we have a future map then set the sharding function
      if (redop == 0)
      {
#ifdef DEBUG_LEGION
        assert(future_map.impl != NULL);
        ReplFutureMapImpl *impl = 
          dynamic_cast<ReplFutureMapImpl*>(future_map.impl);
        assert(impl != NULL);
#else
        ReplFutureMapImpl *impl = 
          static_cast<ReplFutureMapImpl*>(future_map.impl);
#endif
        impl->set_sharding_function(sharding_function);
      }
      // Now we can do the normal prepipeline stage
      IndexTask::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::select_sharding_function(ReplicateContext *repl_ctx)
    //--------------------------------------------------------------------------
    {
      // Do the mapper call to get the sharding function to use
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id); 
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      output.chosen_functor = UINT_MAX;
      mapper->invoke_task_select_sharding_functor(this, input, &output);
      if (output.chosen_functor == UINT_MAX)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s failed to pick a valid sharding functor for "
                      "task %s (UID %lld)", mapper->get_mapper_name(),
                      get_task_name(), get_unique_id())
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Compute the local index space of points for this shard
      if (sharding_space.exists())
        internal_space = 
          sharding_function->find_shard_space(repl_ctx->owner_shard->shard_id,
                                              launch_space, sharding_space);
      else
        internal_space =
          sharding_function->find_shard_space(repl_ctx->owner_shard->shard_id,
                                          launch_space, launch_space->handle);
      // If we're recording then record the local_space
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert(!is_remote());
        assert((tpl != NULL) && tpl->is_recording());
#endif
        tpl->record_local_space(trace_local_id, internal_space);
        // Record the sharding function if needed for the future map
        if (redop == 0)
          tpl->record_sharding_function(trace_local_id, sharding_function);
      }
      // If it's empty we're done, otherwise we go back on the queue
      if (!internal_space.exists())
      {
#ifdef LEGION_SPY
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution(prepare_index_task_complete());
        trigger_children_complete();
        trigger_children_committed();
      }
      else // We have valid points, so it goes on the ready queue
      {
        // Update the total number of points we're actually repsonsible
        // for now with this shard
        IndexSpaceNode *node = runtime->forest->get_node(internal_space);
        total_points = node->get_volume();
#ifdef DEBUG_LEGION
        assert(total_points > 0);
#endif
        enqueue_ready_operation();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl != NULL);
      assert(sharding_collective != NULL);
      sharding_collective->elide_collective();
#endif
      internal_space = tpl->find_local_space(trace_local_id);
      if (redop == 0)
      {
        sharding_function = tpl->find_sharding_function(trace_local_id);
#ifdef DEBUG_LEGION
        assert(future_map.impl != NULL);
        ReplFutureMapImpl *impl = 
          dynamic_cast<ReplFutureMapImpl*>(future_map.impl);
        assert(impl != NULL);
#else
        ReplFutureMapImpl *impl = 
          static_cast<ReplFutureMapImpl*>(future_map.impl);
#endif
        impl->set_sharding_function(sharding_function);
      }
      // If it's empty we're done, otherwise we do the replay
      if (!internal_space.exists())
      {
        // Still have to do this for legion spy
        if (runtime->legion_spy_enabled)
        {
          for (unsigned idx = 0; idx < regions.size(); idx++)
            TaskOp::log_requirement(unique_op_id, idx, regions[idx]);
          runtime->forest->log_launch_space(launch_space->handle, unique_op_id);
        }
#ifdef LEGION_SPY
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution(prepare_index_task_complete());
        resolve_speculation();
        trigger_children_complete();
        trigger_children_committed();
      }
      else
        IndexTask::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
      {
        RegionRequirement &req = logical_regions[idx];
        ProjectionInfo projection_info(runtime, req, launch_space,
                                       sharding_function, sharding_space);
        runtime->forest->perform_dependence_analysis(this, idx, req, 
                                                     projection_info,
                                                     privilege_paths[idx],
                                                     map_applied_conditions);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent ReplIndexTask::prepare_index_task_complete(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      // Set the future if we actually ran the task or we speculated
      if ((redop > 0) && 
          ((speculation_state != RESOLVE_FALSE_STATE) || false_guard.exists()))
      {
        // First time through so start the exchange
        if (deterministic_redop)
        {
          // We have to do the fold of our values here now before
          // we can send them all remotely to the other nodes
          for (std::map<DomainPoint,std::pair<void*,size_t> >::const_iterator
                it = temporary_futures.begin();
                it != temporary_futures.end(); it++)
          {
            fold_reduction_future(it->second.first, it->second.second,
                                  false/*owner*/, true/*exclusive*/);
            legion_free(FUTURE_RESULT_ALLOC, 
                        it->second.first, it->second.second);
          }
          // Clear these out so we don't apply them twice when 
          // we call the base-class version of this method
          temporary_futures.clear();
        }
        // The collective takes ownership of the buffer here
        const RtEvent futures_ready = 
          reduction_collective->exchange_futures(reduction_state);
        // Reinitialize the reduction state buffer so
        // that all the shards can be applied to it in the same order 
        // so that we have bit equivalence across the shards
        reduction_state = NULL;
        initialize_reduction_state();
        if (futures_ready.exists())
          preconditions.insert(futures_ready);
      }
      if ((output_size_collective != NULL) &&
          ((speculation_state != RESOLVE_FALSE_STATE) || false_guard.exists()))
      {
        // Make a copy of the output sizes before we perform all-gather
        local_output_sizes = all_output_sizes;
        // We need to gather output region sizes from all the other shards
        // to determine the sizes of globally indexed output regions
        const RtEvent ready = output_size_collective->exchange_output_sizes();
        if (ready.exists())
          preconditions.insert(ready);
      }
      if (preconditions.empty())
        return RtEvent::NO_RT_EVENT;
      return Runtime::merge_events(preconditions);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      // If we have a reduction operator finalize the exchange of future results
      if (redop > 0)
      {
#ifdef DEBUG_LEGION
        assert(reduction_collective != NULL);
#endif
        reduction_collective->reduce_futures(this);
      }
      // Then we do the base class thing
      IndexTask::trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      // If we already launched then we can just return
      if (launched)
        return;
      // Otherwise, we need to update the internal space so we only set
      // our local points with the predicate false result
      if (redop == 0)
      {
#ifdef DEBUG_LEGION
        ReplicateContext *repl_ctx = 
          dynamic_cast<ReplicateContext*>(parent_ctx);
        assert(repl_ctx != NULL);
#else
        ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
        if (sharding_function == NULL)
        {
          select_sharding_function(repl_ctx);
#ifdef DEBUG_LEGION
          assert(future_map.impl != NULL);
          ReplFutureMapImpl *impl = 
            dynamic_cast<ReplFutureMapImpl*>(future_map.impl);
          assert(impl != NULL);
#else
          ReplFutureMapImpl *impl = 
            static_cast<ReplFutureMapImpl*>(future_map.impl);
#endif
          impl->set_sharding_function(sharding_function);
        }
        // Compute the local index space of points for this shard
        if (sharding_space.exists())
          internal_space = 
            sharding_function->find_shard_space(repl_ctx->owner_shard->shard_id,
                                                launch_space, sharding_space);
        else
          internal_space =
            sharding_function->find_shard_space(repl_ctx->owner_shard->shard_id,
                                            launch_space, launch_space->handle);
      }
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        sharding_collective->elide_collective();
      if (output_size_collective != NULL)
        output_size_collective->elide_collective();
#endif
      // Now continue through and do the base case
      IndexTask::resolve_false(speculated, launched);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_collective == NULL);
#endif
      // If we have a reduction op then we need an exchange
      if (redop > 0)
        reduction_collective = 
          new FutureExchange(ctx, reduction_state_size, COLLECTIVE_LOC_53);
      bool has_output_region = false;
      for (unsigned idx = 0; idx < output_regions.size(); ++idx)
        if (!output_region_options[idx].valid_requirement())
        {
          has_output_region = true;
          break;
        }
      if (has_output_region)
        output_size_collective =
          new OutputSizeExchange(ctx, COLLECTIVE_LOC_29, all_output_sizes);
    } 

    //--------------------------------------------------------------------------
    void ReplIndexTask::set_sharding_function(ShardingID functor,
                                              ShardingFunction *function)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch != NULL);
      assert(sharding_function == NULL);
#endif
      sharding_functor = functor;
      sharding_function = function;
    }

    //--------------------------------------------------------------------------
    FutureMapImpl* ReplIndexTask::create_future_map(TaskContext *ctx,
                                IndexSpace launch_space, IndexSpace shard_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!future_map_ready.exists() || future_map_ready.has_triggered());
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(ctx);
#endif
      Domain shard_domain;
      if (shard_space.exists() && (launch_space != shard_space))
        runtime->forest->find_launch_space_domain(shard_space, shard_domain);
      else
        shard_domain = index_domain;
      future_map_ready = Runtime::create_rt_user_event();
      // Make a replicate future map 
      return new ReplFutureMapImpl(repl_ctx, this,future_map_ready,index_domain, 
          shard_domain, runtime, runtime->get_available_distributed_id(), 
          runtime->address_space);
    }

    //--------------------------------------------------------------------------
    RtEvent ReplIndexTask::find_intra_space_dependence(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {

      AutoLock o_lock(op_lock);
      // Check to see if we already have it
      std::map<DomainPoint,RtEvent>::const_iterator finder = 
        intra_space_dependences.find(point);
      if (finder != intra_space_dependences.end())
        return finder->second;  
      // Make a temporary event and then do different things depending on 
      // whether we own this point or whether a remote shard owns it
      const RtUserEvent pending_event = Runtime::create_rt_user_event();
      intra_space_dependences[point] = pending_event;
      // If not, check to see if this is a point that we expect to own
#ifdef DEBUG_LEGION
      assert(sharding_function != NULL);
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      Domain launch_domain;
      if (sharding_space.exists())
        runtime->forest->find_launch_space_domain(sharding_space,launch_domain);
      else
        launch_space->get_launch_space_domain(launch_domain);
      const ShardID point_shard = 
        sharding_function->find_owner(point, launch_domain); 
      if (point_shard != repl_ctx->owner_shard->shard_id)
      {
        // A different shard owns it so send a message to that shard 
        // requesting it to fill in the dependence
        Serializer rez;
        rez.serialize(repl_ctx->shard_manager->repl_id);
        rez.serialize(point_shard);
        rez.serialize(context_index);
        rez.serialize(point);
        rez.serialize(pending_event);
        rez.serialize(repl_ctx->owner_shard->shard_id);
        repl_ctx->shard_manager->send_intra_space_dependence(point_shard, rez);
      }
      else // We own it so do the normal thing
        pending_intra_space_dependences[point] = pending_event;
      return pending_event; 
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::record_intra_space_dependence(const DomainPoint &point,
                                  const DomainPoint &next, RtEvent point_mapped)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(sharding_function != NULL);
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Determine if the next point is one that we own or is one that is
      // going to be coming from a remote shard
      Domain launch_domain;
      if (sharding_space.exists())
        runtime->forest->find_launch_space_domain(sharding_space,launch_domain);
      else
        launch_space->get_launch_space_domain(launch_domain);
      const ShardID next_shard = 
        sharding_function->find_owner(next, launch_domain); 
      if (next_shard != repl_ctx->owner_shard->shard_id)
      {
        // Make sure we only send this to the repl_ctx once for each 
        // unique shard ID that we see for this point task
        const std::pair<DomainPoint,ShardID> key(point, next_shard); 
        bool record_dependence = true;
        {
          AutoLock o_lock(op_lock);
          std::set<std::pair<DomainPoint,ShardID> >::const_iterator finder = 
            unique_intra_space_deps.find(key);
          if (finder != unique_intra_space_deps.end())
            record_dependence = false;
          else
            unique_intra_space_deps.insert(key);
        }
        if (record_dependence)
          repl_ctx->record_intra_space_dependence(context_index, point, 
                                                  point_mapped, next_shard);
      }
      else // The next shard is ourself, so we can do the normal thing
        IndexTask::record_intra_space_dependence(point, next, point_mapped);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finalize_output_regions(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      ShardMapping *shard_mapping = &repl_ctx->shard_manager->get_mapping();
      RegionTreeForest *forest = runtime->forest;

      for (unsigned idx = 0; idx < output_regions.size(); ++idx)
      {
        const OutputOptions &options = output_region_options[idx];
        if (options.valid_requirement())
          continue;
        IndexSpaceNode *parent = forest->get_node(
            output_regions[idx].parent.get_index_space());
        if (options.global_indexing())
        {
          // For globally indexed output regions, we need a prefix sum to get
          // the right size for each subregion.
          coord_t sum = 0;
          typedef std::map<DomainPoint,size_t> SizeMap;
#ifdef DEBUG_LEGION
          assert(all_output_sizes.find(idx) != all_output_sizes.end());
#endif
          const SizeMap &output_sizes = all_output_sizes[idx];
          const SizeMap &local_sizes = local_output_sizes[idx];
          IndexPartNode *part = runtime->forest->get_node(
            output_regions[idx].partition.get_index_partition());
          for (SizeMap::const_iterator it = output_sizes.begin();
               it != output_sizes.end(); ++it)
          {
            const size_t size = it->second;
            // Make sure we initialize nodes owned by this shard.
            if (local_sizes.find(it->first) != local_sizes.end())
            {
              const LegionColor color =
                part->color_space->linearize_color(it->first);
              IndexSpaceNode *child = part->get_child(color);
              forest->set_pending_space_domain(child->handle,
                  Rect<1>(sum, sum + size - 1), runtime->address_space);
            }
            sum += size;
          }
          log_index.debug(
              "[Task %s (UID: %lld)] setting [0, %lld) to index space %x",
              get_task_name(), get_unique_op_id(), sum, parent->handle.get_id());
          if (parent->set_domain(Rect<1>(0, sum - 1), runtime->address_space,
                                 shard_mapping))
            delete parent;
        }
        // For locally indexed output regions, sizes of subregions are already
        // set when they are fianlized by the point tasks. So we only need to
        // initialize the root index space by taking a union of subspaces.
        else if (parent->set_output_union(all_output_sizes[idx],
                          runtime->address_space, shard_mapping))
          delete parent;
      }
    }

    /////////////////////////////////////////////////////////////
    // Repl Merge Close Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplMergeCloseOp::ReplMergeCloseOp(Runtime *rt)
      : MergeCloseOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplMergeCloseOp::ReplMergeCloseOp(const ReplMergeCloseOp &rhs)
      : MergeCloseOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplMergeCloseOp::~ReplMergeCloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplMergeCloseOp& ReplMergeCloseOp::operator=(const ReplMergeCloseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplMergeCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_close();
      mapped_barrier = RtBarrier::NO_RT_BARRIER;
    }

    //--------------------------------------------------------------------------
    void ReplMergeCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_close();
      runtime->free_repl_merge_close_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplMergeCloseOp::set_repl_close_info(RtBarrier mapped)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!mapped_barrier.exists());
#endif
      mapped_barrier = mapped;
    }

    //--------------------------------------------------------------------------
    void ReplMergeCloseOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapped_barrier.exists());
      assert(mapping_tracker != NULL);
#endif
      // All we have to do is add our map precondition to the tracker
      // so we know we are mapping in order with respect to other
      // repl close operations that use the same close index
      mapping_tracker->add_mapping_dependence(
          mapped_barrier.get_previous_phase());
    }

    //--------------------------------------------------------------------------
    void ReplMergeCloseOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapped_barrier.exists());
#endif
      // Arrive on our barrier with the precondition
      Runtime::phase_barrier_arrive(mapped_barrier, 1/*count*/);
      // Then complete the mapping once the barrier has triggered
      complete_mapping(mapped_barrier);
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Repl Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplFillOp::ReplFillOp(Runtime *rt)
      : FillOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplFillOp::ReplFillOp(const ReplFillOp &rhs)
      : FillOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplFillOp::~ReplFillOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplFillOp& ReplFillOp::operator=(const ReplFillOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
      mapped_collective_id = 
        ctx->get_next_collective_index(COLLECTIVE_LOC_2);
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_fill();
      sharding_functor = UINT_MAX;
      sharding_function = NULL;
      mapper = NULL;
#ifdef DEBUG_LEGION
      sharding_collective = NULL;
#endif
      mapped_collective_id = UINT_MAX;
      mapped_collective = NULL;
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      if (mapped_collective != NULL)
        delete mapped_collective;
      deactivate_fill();
      runtime->free_repl_fill_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Do the mapper call to get the sharding function to use
      if (mapper == NULL)
        mapper = runtime->find_mapper(
            parent_ctx->get_executing_processor(), map_id); 
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      output.chosen_functor = UINT_MAX; 
      mapper->invoke_fill_select_sharding_functor(this, input, &output);
      if (output.chosen_functor == UINT_MAX)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s failed to pick a valid sharding functor for "
                      "fill in task %s (UID %lld)", mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
#ifdef DEBUG_LEGION
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s chose different sharding functions "
                      "for fill in task %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id())
#endif
      // Now we can do the normal prepipeline stage
      FillOp::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Figure out whether this shard owns this point
      ShardID owner_shard;
      if (sharding_space.exists())
      {
        Domain shard_domain;
        runtime->forest->find_launch_space_domain(sharding_space, shard_domain);
        owner_shard = sharding_function->find_owner(index_point, shard_domain);
      }
      else
        owner_shard = sharding_function->find_owner(index_point, index_domain); 
      // If we're recording then record the owner shard
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert((tpl != NULL) && tpl->is_recording());
#endif
        tpl->record_owner_shard(trace_local_id, owner_shard);
      }
      if (runtime->legion_spy_enabled)
        LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
#ifdef DEBUG_LEGION
      assert(mapped_collective == NULL);
#endif
      mapped_collective = 
        new ShardEventTree(repl_ctx, owner_shard, mapped_collective_id);
      // If we own it we go on the queue, otherwise we complete early
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
#ifdef LEGION_SPY
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We don't own it, so we can pretend like we
        // mapped and executed this fill already
        // Before we do this though we have to get the version state
        // names for any writes so we can update our local state
        RtEvent local_done = mapped_collective->get_local_event();
        complete_mapping(local_done);
        complete_execution();
      }
      else // We own it, so do the base call
      {
        // Signal the tree when we are done our mapping
        mapped_collective->signal_tree(mapped_event);
        // Perform the base operation 
        FillOp::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl != NULL);
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
      assert(sharding_collective != NULL);
      sharding_collective->elide_collective();
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      const ShardID owner_shard = tpl->find_owner_shard(trace_local_id);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        // Still have to do this for legion spy
        if (runtime->legion_spy_enabled && !need_prepipeline_stage)
          log_fill_requirement();
#ifdef LEGION_SPY
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        complete_mapping();
        complete_execution();
        resolve_speculation();
      }
      else // We own it, so do the base call
        FillOp::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplFillOp::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      if (launched)
        return;
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        sharding_collective->elide_collective();
#endif
      FillOp::resolve_false(speculated, launched);
    }

    /////////////////////////////////////////////////////////////
    // Repl Index Fill Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndexFillOp::ReplIndexFillOp(Runtime *rt)
      : IndexFillOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplIndexFillOp::ReplIndexFillOp(const ReplIndexFillOp &rhs)
      : IndexFillOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplIndexFillOp::~ReplIndexFillOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplIndexFillOp& ReplIndexFillOp::operator=(const ReplIndexFillOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_index_fill();
      sharding_functor = UINT_MAX;
      sharding_function = NULL;
      mapper = NULL;
#ifdef DEBUG_LEGION
      sharding_collective = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      deactivate_index_fill();
      runtime->free_repl_index_fill_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Do the mapper call to get the sharding function to use
      if (mapper == NULL)
        mapper = runtime->find_mapper(
            parent_ctx->get_executing_processor(), map_id);
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      output.chosen_functor = UINT_MAX;
      mapper->invoke_fill_select_sharding_functor(this, input, &output);
      if (output.chosen_functor == UINT_MAX)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s failed to pick a valid sharding functor for "
                      "index fill in task %s (UID %lld)", 
                      mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
#ifdef DEBUG_LEGION
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s chose different sharding functions "
                      "for index fill in task %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
#endif
      // Now we can do the normal prepipeline stage
      IndexFillOp::trigger_prepipeline_stage();
    }
    
    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      ProjectionInfo projection_info(runtime, requirement, launch_space, 
                                     sharding_function, sharding_space);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
      assert(launch_space != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Compute the local index space of points for this shard
      IndexSpace local_space;
      if (sharding_space.exists())
        local_space =
          sharding_function->find_shard_space(repl_ctx->owner_shard->shard_id, 
                                              launch_space, sharding_space);
      else
        local_space =
          sharding_function->find_shard_space(repl_ctx->owner_shard->shard_id, 
                                          launch_space, launch_space->handle);
      // If we're recording then record the local_space
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert((tpl != NULL) && tpl->is_recording());
#endif
        tpl->record_local_space(trace_local_id, local_space);
      }
      // If it's empty we're done, otherwise we go back on the queue
      if (!local_space.exists())
      {
#ifdef LEGION_SPY
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution();
      }
      else // We have valid points, so it goes on the ready queue
      {
        if (remove_launch_space_reference(launch_space))
          delete launch_space;
        launch_space = runtime->forest->get_node(local_space);
        add_launch_space_reference(launch_space);
        IndexFillOp::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl != NULL);
      assert(sharding_collective != NULL);
      sharding_collective->elide_collective();
#endif
      const IndexSpace local_space = tpl->find_local_space(trace_local_id);
      // If it's empty we're done, otherwise we do the replay
      if (!local_space.exists())
      {
        // Still have to do this for legion spy
        if (runtime->legion_spy_enabled)
          log_index_fill_requirement();
#ifdef LEGION_SPY
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution();
        resolve_speculation();
      }
      else
      {
        if (remove_launch_space_reference(launch_space))
          delete launch_space;
        launch_space = runtime->forest->get_node(local_space);
        add_launch_space_reference(launch_space);
        IndexFillOp::trigger_replay();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      if (launched)
        return;
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        sharding_collective->elide_collective();
#endif
      IndexFillOp::resolve_false(speculated, launched);
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Repl Copy Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplCopyOp::ReplCopyOp(Runtime *rt)
      : CopyOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplCopyOp::ReplCopyOp(const ReplCopyOp &rhs)
      : CopyOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplCopyOp::~ReplCopyOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplCopyOp& ReplCopyOp::operator=(const ReplCopyOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
      mapped_collective_id = 
        ctx->get_next_collective_index(COLLECTIVE_LOC_2);
      // Initialize our index domain of a single point
      index_domain = Domain(index_point, index_point);
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_copy();
      sharding_functor = UINT_MAX;
      sharding_function = NULL;
#ifdef DEBUG_LEGION
      sharding_collective = NULL;
#endif
      mapped_collective_id = UINT_MAX;
      mapped_collective = NULL;
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      if (mapped_collective != NULL)
        delete mapped_collective;
      deactivate_copy();
      runtime->free_repl_copy_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Do the mapper call to get the sharding function to use
      if (mapper == NULL)
        mapper = runtime->find_mapper(
            parent_ctx->get_executing_processor(), map_id); 
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      output.chosen_functor = UINT_MAX; 
      mapper->invoke_copy_select_sharding_functor(this, input, &output);
      if (output.chosen_functor == UINT_MAX)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s failed to pick a valid sharding functor for "
                      "copy in task %s (UID %lld)", mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
#ifdef DEBUG_LEGION
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s chose different sharding functions "
                      "for copy in task %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id())
#endif
      // Now we can do the normal prepipeline stage
      CopyOp::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Figure out whether this shard owns this point
      ShardID owner_shard;
      if (sharding_space.exists())
      {
        Domain shard_domain;
        runtime->forest->find_launch_space_domain(sharding_space, shard_domain);
        owner_shard = sharding_function->find_owner(index_point, shard_domain);
      }
      else
        owner_shard = sharding_function->find_owner(index_point, index_domain); 
      // If we're recording then record the owner shard
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert((tpl != NULL) && tpl->is_recording());
#endif
        tpl->record_owner_shard(trace_local_id, owner_shard);
      }
      if (runtime->legion_spy_enabled)
        LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
#ifdef DEBUG_LEGION
      assert(mapped_collective == NULL);
#endif
      mapped_collective = 
        new ShardEventTree(repl_ctx, owner_shard, mapped_collective_id);
      // If we own it we go on the queue, otherwise we complete early
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
#ifdef LEGION_SPY
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We don't own it, so we can pretend like we
        // mapped and executed this copy already
        // Before we do this though we have to get the version state
        // names for any writes so we can update our local state
        RtEvent local_done = mapped_collective->get_local_event();
        complete_mapping(local_done);
        complete_execution();
      }
      else // We own it, so do the base call
      {
        // Signal the tree when we are done our mapping
        mapped_collective->signal_tree(mapped_event);
        // Perform the base operation 
        CopyOp::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl != NULL);
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
      assert(sharding_collective != NULL);
      sharding_collective->elide_collective();
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      const ShardID owner_shard = tpl->find_owner_shard(trace_local_id);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        // Still have to do this for legion spy
        if (runtime->legion_spy_enabled && !need_prepipeline_stage)
          log_copy_requirements();
#ifdef LEGION_SPY
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        complete_mapping();
        complete_execution();
        resolve_speculation();
      }
      else // We own it, so do the base call
        CopyOp::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      if (launched)
        return;
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        sharding_collective->elide_collective();
#endif
      CopyOp::resolve_false(speculated, launched);
    }

    /////////////////////////////////////////////////////////////
    // Repl Index Copy Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndexCopyOp::ReplIndexCopyOp(Runtime *rt)
      : IndexCopyOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplIndexCopyOp::ReplIndexCopyOp(const ReplIndexCopyOp &rhs)
      : IndexCopyOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplIndexCopyOp::~ReplIndexCopyOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplIndexCopyOp& ReplIndexCopyOp::operator=(const ReplIndexCopyOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_index_copy();
      sharding_functor = UINT_MAX;
      sharding_function = NULL;
#ifdef DEBUG_LEGION
      sharding_collective = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      indirection_barriers.clear();
      if (!src_collectives.empty())
      {
        for (unsigned idx = 0; idx < src_collectives.size(); idx++)
          delete src_collectives[idx];
        src_collectives.clear();
      }
      if (!dst_collectives.empty())
      {
        for (unsigned idx = 0; idx < dst_collectives.size(); idx++)
          delete dst_collectives[idx];
        dst_collectives.clear();
      }
      deactivate_index_copy();
      runtime->free_repl_index_copy_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Do the mapper call to get the sharding function to use
      if (mapper == NULL)
        mapper = runtime->find_mapper(
            parent_ctx->get_executing_processor(), map_id); 
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      output.chosen_functor = UINT_MAX;
      mapper->invoke_copy_select_sharding_functor(this, input, &output);
      if (output.chosen_functor == UINT_MAX)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s failed to pick a valid sharding functor for "
                      "index copy in task %s (UID %lld)", 
                      mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor); 
#ifdef DEBUG_LEGION
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s chose different sharding functions "
                      "for index copy in task %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
#endif
      // Now we can do the normal prepipeline stage
      IndexCopyOp::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      for (unsigned idx = 0; idx < src_requirements.size(); idx++)
      {
        ProjectionInfo projection_info (runtime, src_requirements[idx], 
                       launch_space, sharding_function, sharding_space);
        runtime->forest->perform_dependence_analysis(this, idx, 
                                                     src_requirements[idx],
                                                     projection_info,
                                                     src_privilege_paths[idx],
                                                     map_applied_conditions);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        ProjectionInfo projection_info(runtime, dst_requirements[idx], 
                       launch_space, sharding_function, sharding_space);
        unsigned index = src_requirements.size()+idx;
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_READ_WRITE;
        runtime->forest->perform_dependence_analysis(this, index, 
                                                     dst_requirements[idx],
                                                     projection_info,
                                                     dst_privilege_paths[idx],
                                                     map_applied_conditions);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = LEGION_REDUCE;
      }
      if (!src_indirect_requirements.empty())
      {
        gather_versions.resize(src_indirect_requirements.size());
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
        {
          ProjectionInfo gather_info(runtime, src_indirect_requirements[idx], 
                             launch_space, sharding_function, sharding_space);
          runtime->forest->perform_dependence_analysis(this, idx, 
                                                 src_indirect_requirements[idx],
                                                 gather_info,
                                                 gather_privilege_paths[idx],
                                                 map_applied_conditions);
        }
      }
      if (!dst_indirect_requirements.empty())
      {
        scatter_versions.resize(dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
        {
          ProjectionInfo scatter_info(runtime, dst_indirect_requirements[idx],
                              launch_space, sharding_function, sharding_space);
          runtime->forest->perform_dependence_analysis(this, idx, 
                                                 dst_indirect_requirements[idx],
                                                 scatter_info,
                                                 scatter_privilege_paths[idx],
                                                 map_applied_conditions);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Compute the local index space of points for this shard
      IndexSpace local_space;
      if (sharding_space.exists())
        local_space =
          sharding_function->find_shard_space(repl_ctx->owner_shard->shard_id,
                                              launch_space, sharding_space);
      else
        local_space =
          sharding_function->find_shard_space(repl_ctx->owner_shard->shard_id,
                                          launch_space, launch_space->handle);
      // If we're recording then record the local_space
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert((tpl != NULL) && tpl->is_recording());
#endif
        tpl->record_local_space(trace_local_id, local_space);
      }
      // If it's empty we're done, otherwise we go back on the queue
      if (!local_space.exists())
      {
        // If we have indirections then we still need to participate in those
        if (!src_indirect_requirements.empty() &&
            collective_src_indirect_points)
        {
          LegionVector<IndirectRecord>::aligned empty_records;
          for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          {
            src_collectives[idx]->exchange_records(empty_records);
            empty_records.clear();
          }
        }
        if (!dst_indirect_requirements.empty() && 
            collective_dst_indirect_points)
        {
          LegionVector<IndirectRecord>::aligned empty_records;
          for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          {
            dst_collectives[idx]->exchange_records(empty_records);
            empty_records.clear();
          }
        }
        // Arrive on our indirection barriers if we have them
        if (!indirection_barriers.empty())
        {
          for (unsigned idx = 0; idx < indirection_barriers.size(); idx++)
            Runtime::phase_barrier_arrive(indirection_barriers[idx],1/*count*/);
        }
#ifdef LEGION_SPY
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution();
      }
      else // If we have any valid points do the base call
      {
        if (remove_launch_space_reference(launch_space))
          delete launch_space;
        launch_space = runtime->forest->get_node(local_space);
        add_launch_space_reference(launch_space);
        IndexCopyOp::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(tpl != NULL);
      assert(sharding_collective != NULL);
      sharding_collective->elide_collective();
#endif
      const IndexSpace local_space = tpl->find_local_space(trace_local_id);
      // If it's empty we're done, otherwise we do the replay
      if (!local_space.exists())
      {
        // Still have to do this for legion spy
        if (runtime->legion_spy_enabled && !need_prepipeline_stage)
          log_index_copy_requirements();
#ifdef LEGION_SPY
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(unique_op_id, 
            ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution();
        resolve_speculation();
      }
      else
      {
        if (remove_launch_space_reference(launch_space))
          delete launch_space;
        launch_space = runtime->forest->get_node(local_space);
        add_launch_space_reference(launch_space);
        IndexCopyOp::trigger_replay();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::resolve_false(bool speculated, bool launched)
    //--------------------------------------------------------------------------
    {
      if (launched)
        return;
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        sharding_collective->elide_collective();
#endif
      IndexCopyOp::resolve_false(speculated, launched);
    }

    //--------------------------------------------------------------------------
    ApEvent ReplIndexCopyOp::exchange_indirect_records(const unsigned index,
             const ApEvent local_done, const PhysicalTraceInfo &trace_info,
             const InstanceSet &instances, const IndexSpace space,
             const DomainPoint &key,
             LegionVector<IndirectRecord>::aligned &records, const bool sources)
    //--------------------------------------------------------------------------
    {
      if (sources && !collective_src_indirect_points)
        return CopyOp::exchange_indirect_records(index, local_done, trace_info,
                                        instances, space, key, records, sources);
      if (!sources && !collective_dst_indirect_points)
        return CopyOp::exchange_indirect_records(index, local_done, trace_info,
                                        instances, space, key, records, sources);
#ifdef DEBUG_LEGION
      assert(local_done.exists());
      assert(index < indirection_barriers.size());
      assert(indirection_barriers[index].exists());
#endif
      RtEvent wait_on;
      RtUserEvent to_trigger;
      std::set<ApEvent> arrival_events;
      {
        IndexSpaceNode *node = runtime->forest->get_node(space);
        ApEvent domain_ready;
        const Domain dom = node->get_domain(domain_ready, true/*tight*/);
        // Take the lock and record our sets and instances
        AutoLock o_lock(op_lock);
        if (sources)
        {
          if (domain_ready.exists() && !domain_ready.has_triggered())
          {
            for (unsigned idx = 0; idx < instances.size(); idx++)
            {
              const InstanceRef &ref = instances[idx];
              const ApEvent inst_ready = ref.get_ready_event();
              if (inst_ready.exists() && !inst_ready.has_triggered())
                src_records[index].push_back(IndirectRecord(
                      ref.get_valid_fields(), ref.get_manager(), key, space,
                      Runtime::merge_events(&trace_info, domain_ready,
                        inst_ready), dom));
              else
                src_records[index].push_back(IndirectRecord(
                      ref.get_valid_fields(), ref.get_manager(), key,
                      space, domain_ready, dom));
            }
          }
          else
          {
            for (unsigned idx = 0; idx < instances.size(); idx++)
            {
              const InstanceRef &ref = instances[idx];
              src_records[index].push_back(IndirectRecord(
                    ref.get_valid_fields(), ref.get_manager(), key,
                    space, ref.get_ready_event(), dom));
            }
          }
          src_exchange_events[index].insert(local_done);
          if (!src_exchanged[index].exists())
            src_exchanged[index] = Runtime::create_rt_user_event();
          if (src_exchange_events[index].size() == points.size())
          {
            to_trigger = src_exchanged[index];
            arrival_events.insert(src_exchange_events[index].begin(),
                src_exchange_events[index].end());
          }
          else
            wait_on = src_exchanged[index];
        }
        else
        {
          if (domain_ready.exists() && !domain_ready.has_triggered())
          {
            for (unsigned idx = 0; idx < instances.size(); idx++)
            {
              const InstanceRef &ref = instances[idx];
              const ApEvent inst_ready = ref.get_ready_event();
              if (inst_ready.exists() && !inst_ready.has_triggered())
                dst_records[index].push_back(IndirectRecord(
                      ref.get_valid_fields(), ref.get_manager(), key, space,
                      Runtime::merge_events(&trace_info, domain_ready,
                        inst_ready), dom));
              else
                dst_records[index].push_back(IndirectRecord(
                      ref.get_valid_fields(), ref.get_manager(), key,
                      space, domain_ready, dom));
            }
          }
          else
          {
            for (unsigned idx = 0; idx < instances.size(); idx++)
            {
              const InstanceRef &ref = instances[idx];
              dst_records[index].push_back(IndirectRecord(
                    ref.get_valid_fields(), ref.get_manager(), key,
                    space, ref.get_ready_event(), dom));
            }
          }
          dst_exchange_events[index].insert(local_done);
          if (!dst_exchanged[index].exists())
            dst_exchanged[index] = Runtime::create_rt_user_event();
          if (dst_exchange_events[index].size() == points.size())
          {
            to_trigger = dst_exchanged[index];
            arrival_events.insert(dst_exchange_events[index].begin(),
                dst_exchange_events[index].end());
          }
          else
            wait_on = dst_exchanged[index];
        }
      }
      if (to_trigger.exists())
      {
        // Perform the collective
        if (sources)
          src_collectives[index]->exchange_records(src_records[index]);
        else
          dst_collectives[index]->exchange_records(dst_records[index]);
        Runtime::trigger_event(to_trigger);
        if (!arrival_events.empty())
          Runtime::phase_barrier_arrive(indirection_barriers[index],
              1/*count*/, Runtime::merge_events(&trace_info, arrival_events));
      }
      else if (!wait_on.has_triggered())
        wait_on.wait();
      // Once we wake up we can copy out the results
      if (sources)
        records = src_records[index];
      else
        records = dst_records[index];
      return indirection_barriers[index];
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::initialize_replication(ReplicateContext *ctx,
                          std::vector<ApBarrier> &indirection_bars, 
                          unsigned &next_indirection_index)
    //--------------------------------------------------------------------------
    { 
      if (!src_indirect_requirements.empty() && collective_src_indirect_points)
      {
        src_collectives.resize(src_indirect_requirements.size());
        for (unsigned idx = 0; idx < src_indirect_requirements.size(); idx++)
          src_collectives[idx] = 
            new IndirectRecordExchange(ctx, COLLECTIVE_LOC_80);
      }
      if (!dst_indirect_requirements.empty() && collective_dst_indirect_points)
      {
        dst_collectives.resize(dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < dst_indirect_requirements.size(); idx++)
          dst_collectives[idx] = 
            new IndirectRecordExchange(ctx, COLLECTIVE_LOC_81);
      }
      if (!src_indirect_requirements.empty() || 
          !dst_indirect_requirements.empty())
      {
#ifdef DEBUG_LEGION
        assert(src_indirect_requirements.empty() ||
               dst_indirect_requirements.empty() ||
               (src_indirect_requirements.size() == 
                dst_indirect_requirements.size()));
#endif
        indirection_barriers.resize(
            (src_indirect_requirements.size() > 
              dst_indirect_requirements.size()) ?
                src_indirect_requirements.size() : 
                dst_indirect_requirements.size());
        for (unsigned idx = 0; idx < indirection_barriers.size(); idx++)
        {
          ApBarrier &next_bar = indirection_bars[next_indirection_index++]; 
          indirection_barriers[idx] = next_bar;
          ctx->advance_replicate_barrier(next_bar, ctx->total_shards);
          if (next_indirection_index == indirection_bars.size())
            next_indirection_index = 0;
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Repl Deletion Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplDeletionOp::ReplDeletionOp(Runtime *rt)
      : DeletionOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplDeletionOp::ReplDeletionOp(const ReplDeletionOp &rhs)
      : DeletionOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplDeletionOp::~ReplDeletionOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplDeletionOp& ReplDeletionOp::operator=(const ReplDeletionOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_deletion();
      ready_barrier = RtBarrier::NO_RT_BARRIER;
      mapping_barrier = RtBarrier::NO_RT_BARRIER;
      execution_barrier = RtBarrier::NO_RT_BARRIER;
      is_total_sharding = false;
      is_first_local_shard = false;
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_deletion();
      runtime->free_repl_deletion_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if ((kind == FIELD_DELETION) || (kind == LOGICAL_REGION_DELETION))
        Runtime::phase_barrier_arrive(ready_barrier, 1/*count*/);
      if (kind == FIELD_DELETION)
      {
#ifdef DEBUG_LEGION
        ReplicateContext *repl_ctx = 
          dynamic_cast<ReplicateContext*>(parent_ctx);
        assert(repl_ctx != NULL);
#else
        ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
        // Field deletions need to compute their version infos
        if ((is_total_sharding && is_first_local_shard) || 
            (repl_ctx->owner_shard->shard_id == 0)) 
        {
          std::set<RtEvent> preconditions;
          version_infos.resize(deletion_requirements.size());
          for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
            runtime->forest->perform_versioning_analysis(this, idx,
                                              deletion_requirements[idx],
                                              version_infos[idx],
                                              preconditions);
          if (!preconditions.empty())
          {
            preconditions.insert(ready_barrier);
            enqueue_ready_operation(Runtime::merge_events(preconditions));
            return;
          }
        }
      }
      enqueue_ready_operation(ready_barrier);
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(execution_barrier.exists());
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // There are two different implementations here depending on whether we
      // know that we have a deletion operation on every shard or not
      // If not, we just let the deletion for shard 0 do all the work, 
      // otherwise we know we can evenly distribute the work
      if (kind == LOGICAL_REGION_DELETION)
      {
        // Just need to clean out the version managers which will free
        // all the equivalence sets and allow the reference counting to
        // clean everything up
        if (is_first_local_shard)
        {
          bool has_outermost = false;
          RegionTreeContext outermost_ctx;
          const RegionTreeContext tree_context = parent_ctx->get_context();
          for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
          {
            const RegionRequirement &req = deletion_requirements[idx];
            if (returnable_privileges[idx])
            {
              if (!has_outermost)
              {
                TaskContext *outermost = 
                  parent_ctx->find_outermost_local_context();
                outermost_ctx = outermost->get_context();
                has_outermost = true;
              }
              runtime->forest->invalidate_versions(outermost_ctx, req.region);
            }
            else
              runtime->forest->invalidate_versions(tree_context, req.region);
          }
        }
        complete_mapping();
      }
      else if (kind == FIELD_DELETION)
      {
#ifdef DEBUG_LEGION
        assert(mapping_barrier.exists());
#endif
        if ((is_total_sharding && is_first_local_shard) || 
            (repl_ctx->owner_shard->shard_id == 0))
        {
          // For this case we actually need to go through and prune out any
          // valid instances for these fields in the equivalence sets in order
          // to be able to free up the resources.
          const TraceInfo trace_info(this);
          for (unsigned idx = 0; idx < deletion_requirements.size(); idx++)
            runtime->forest->invalidate_fields(this, idx, version_infos[idx],
                PhysicalTraceInfo(trace_info, idx), map_applied_conditions, 
                is_total_sharding/*collective*/);
        }
        // make sure that we don't try to do the deletion calls until
        // after the allocator is ready
        if (allocator->ready_event.exists())
          map_applied_conditions.insert(allocator->ready_event);
        if (!map_applied_conditions.empty())
          Runtime::phase_barrier_arrive(mapping_barrier, 1/*count*/,
              Runtime::merge_events(map_applied_conditions));
        else
          Runtime::phase_barrier_arrive(mapping_barrier, 1/*count*/);
        complete_mapping(mapping_barrier);
      }
      else
        complete_mapping();
      // complete execution once all the shards are done
      if (execution_precondition.exists())
        Runtime::phase_barrier_arrive(execution_barrier, 1/*count*/, 
            Runtime::protect_event(execution_precondition));
      else
        Runtime::phase_barrier_arrive(execution_barrier, 1/*count*/);
      complete_execution(execution_barrier);
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      std::set<RtEvent> applied;
      if (is_total_sharding && is_first_local_shard)
      {
        switch (kind)
        {
          case INDEX_SPACE_DELETION:
            {
#ifdef DEBUG_LEGION
              assert(deletion_req_indexes.empty());
#endif
              runtime->forest->destroy_index_space(index_space,
                                                   applied, true/*collective*/);
              if (!sub_partitions.empty())
              {
                for (std::vector<IndexPartition>::const_iterator it = 
                      sub_partitions.begin(); it != sub_partitions.end(); it++)
                  runtime->forest->destroy_index_partition(*it, applied,
                                                           true/*collective*/);
              }
              break;
            }
          case INDEX_PARTITION_DELETION:
            {
#ifdef DEBUG_LEGION
              assert(deletion_req_indexes.empty());
#endif
              runtime->forest->destroy_index_partition(index_part, applied,
                                                       true/*collective*/);
              if (!sub_partitions.empty())
              {
                for (std::vector<IndexPartition>::const_iterator it = 
                      sub_partitions.begin(); it != sub_partitions.end(); it++)
                  runtime->forest->destroy_index_partition(*it, applied,
                                                           true/*collective*/);
              }
              break;
            }
          case FIELD_SPACE_DELETION:
            {
#ifdef DEBUG_LEGION
              assert(deletion_req_indexes.empty());
#endif
              runtime->forest->destroy_field_space(field_space, applied, 
                                                   true/*collective*/);
              break;
            }
          case FIELD_DELETION:
            // Everyone is going to do the same thing for field deletions
            break;
          case LOGICAL_REGION_DELETION:
            {
              // Only do something here if we don't have any parent req indexes
              // If we had no deletion requirements then we know there is
              // nothing to race with and we can just do our deletion
              if (parent_req_indexes.empty())
                runtime->forest->destroy_logical_region(logical_region, applied,
                                                        true/*collective*/);
              break;
            }
          default:
            assert(false);
        }
      }
      else if (repl_ctx->owner_shard->shard_id == 0)
      {
        // Shard 0 will handle the actual deletions
        switch (kind)
        {
          case INDEX_SPACE_DELETION:
            {
#ifdef DEBUG_LEGION
              assert(deletion_req_indexes.empty());
#endif
              runtime->forest->destroy_index_space(index_space, applied);
              if (!sub_partitions.empty())
              {
                for (std::vector<IndexPartition>::const_iterator it = 
                      sub_partitions.begin(); it != sub_partitions.end(); it++)
                  runtime->forest->destroy_index_partition(*it, applied);
              }
              break;
            }
          case INDEX_PARTITION_DELETION:
            {
#ifdef DEBUG_LEGION
              assert(deletion_req_indexes.empty());
#endif
              runtime->forest->destroy_index_partition(index_part, applied);
              if (!sub_partitions.empty())
              {
                for (std::vector<IndexPartition>::const_iterator it = 
                      sub_partitions.begin(); it != sub_partitions.end(); it++)
                  runtime->forest->destroy_index_partition(*it, applied);
              }
              break;
            }
          case FIELD_SPACE_DELETION:
            {
#ifdef DEBUG_LEGION
              assert(deletion_req_indexes.empty());
#endif
              runtime->forest->destroy_field_space(field_space, applied);
              break;
            }
          case FIELD_DELETION:
            // Everyone is going to do the same thing for field deletions
            break;
          case LOGICAL_REGION_DELETION:
            {
              // Only do something here if we don't have any parent req indexes
              // If we had no deletion requirements then we know there is
              // nothing to race with and we can just do our deletion
              if (parent_req_indexes.empty())
                runtime->forest->destroy_logical_region(logical_region,applied);
              break;
            }
          default:
            assert(false);
        }
      }
      std::vector<LogicalRegion> regions_to_destroy;
      // If this is a field deletion then everyone does the same thing
      if (kind == FIELD_DELETION)
      {
        if (!local_fields.empty())
          runtime->forest->free_local_fields(field_space, local_fields, 
                              local_field_indexes, true/*collective*/);
        if (!global_fields.empty())
          runtime->forest->free_fields(field_space, global_fields, applied, 
                                   (repl_ctx->owner_shard->shard_id != 0));
        parent_ctx->remove_deleted_fields(free_fields, parent_req_indexes);
        if (!local_fields.empty())
          parent_ctx->remove_deleted_local_fields(field_space, local_fields);
        if (!deletion_req_indexes.empty())
          parent_ctx->remove_deleted_requirements(deletion_req_indexes,
                                                  regions_to_destroy);
      }
      else if ((kind == LOGICAL_REGION_DELETION) && !parent_req_indexes.empty())
        parent_ctx->remove_deleted_requirements(parent_req_indexes,
                                                regions_to_destroy);
      if (!regions_to_destroy.empty())
      {
        // Only selectively delete depending on our configuration
        if (is_total_sharding && is_first_local_shard)
        {
          for (std::vector<LogicalRegion>::const_iterator it =
               regions_to_destroy.begin(); it != regions_to_destroy.end(); it++)
            runtime->forest->destroy_logical_region(*it, applied, 
                                                    true/*collective*/);
        }
        else if (repl_ctx->owner_shard->shard_id == 0)
        {
          for (std::vector<LogicalRegion>::const_iterator it =
               regions_to_destroy.begin(); it != regions_to_destroy.end(); it++)
            runtime->forest->destroy_logical_region(*it, applied);
        }
      }
#ifdef LEGION_SPY
      // Still have to do this for legion spy
      LegionSpy::log_operation_events(unique_op_id, 
          ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
      if (!applied.empty())
        complete_operation(Runtime::merge_events(applied));
      else
        complete_operation();
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::initialize_replication(ReplicateContext *ctx,
                                                RtBarrier &delready_barrier,
                                                RtBarrier &delmap_barrier, 
                                                RtBarrier &delexec_barrier,
                                                bool is_total, bool is_first,
                                                bool unordered/*=false*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!ready_barrier.exists());
      assert(!mapping_barrier.exists());
      assert(!execution_barrier.exists());
#endif
      // Only field and region deletions need a ready barrier since they
      // will be touching the physical states of the region tree
      if ((kind == LOGICAL_REGION_DELETION) || (kind == FIELD_DELETION))
      {
        ready_barrier = delready_barrier;
        if (unordered)
          Runtime::advance_barrier(delready_barrier);
        else
          ctx->advance_replicate_barrier(delready_barrier, ctx->total_shards);
        // Only field deletions need a mapping barrier for downward facing
        // dependences in other shards
        if (kind == FIELD_DELETION)
        {
          mapping_barrier = delmap_barrier;
          if (unordered)
            Runtime::advance_barrier(delmap_barrier);
          else
            ctx->advance_replicate_barrier(delmap_barrier, ctx->total_shards);
        }
      }
      // All deletion kinds need an execution barrier
      execution_barrier = delexec_barrier;
      if (unordered)
        Runtime::advance_barrier(delexec_barrier);
      else
        ctx->advance_replicate_barrier(delexec_barrier, ctx->total_shards);
      is_total_sharding = is_total;
      is_first_local_shard = is_first;
    }

    //--------------------------------------------------------------------------
    void ReplDeletionOp::record_unordered_kind(
       std::map<IndexSpace,ReplDeletionOp*> &index_space_deletions,
       std::map<IndexPartition,ReplDeletionOp*> &index_partition_deletions,
       std::map<FieldSpace,ReplDeletionOp*> field_space_deletions,
       std::map<std::pair<FieldSpace,FieldID>,ReplDeletionOp*> &field_deletions,
       std::map<LogicalRegion,ReplDeletionOp*> &logical_region_deletions)
    //--------------------------------------------------------------------------
    {
      switch (kind)
      {
        case INDEX_SPACE_DELETION:
          {
#ifdef DEBUG_LEGION
            assert(index_space_deletions.find(index_space) ==
                    index_space_deletions.end());
#endif
            index_space_deletions[index_space] = this;
            break;
          }
        case INDEX_PARTITION_DELETION:
          {
#ifdef DEBUG_LEGION
            assert(index_partition_deletions.find(index_part) ==
                    index_partition_deletions.end());
#endif
            index_partition_deletions[index_part] = this;
            break;
          }
        case FIELD_SPACE_DELETION:
          {
#ifdef DEBUG_LEGION
            assert(field_space_deletions.find(field_space) ==
                    field_space_deletions.end());
#endif
            field_space_deletions[field_space] = this;
            break;
          }
        case FIELD_DELETION:
          {
#ifdef DEBUG_LEGION
            assert(!free_fields.empty());
#endif
            const std::pair<FieldSpace,FieldID> key(field_space,
                *(free_fields.begin()));
#ifdef DEBUG_LEGION
            assert(field_deletions.find(key) == field_deletions.end());
#endif
            field_deletions[key] = this;
            break;
          }
        case LOGICAL_REGION_DELETION:
          {
#ifdef DEBUG_LEGION
            assert(logical_region_deletions.find(logical_region) ==
                    logical_region_deletions.end());
#endif
            logical_region_deletions[logical_region] = this;
            break;
          }
        default:
          assert(false); // should never get here
      }
    }

    /////////////////////////////////////////////////////////////
    // Repl Pending Partition Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplPendingPartitionOp::ReplPendingPartitionOp(Runtime *rt)
      : PendingPartitionOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplPendingPartitionOp::ReplPendingPartitionOp(
                                              const ReplPendingPartitionOp &rhs)
      : PendingPartitionOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplPendingPartitionOp::~ReplPendingPartitionOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplPendingPartitionOp& ReplPendingPartitionOp::operator=(
                                              const ReplPendingPartitionOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplPendingPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_pending();
    }

    //--------------------------------------------------------------------------
    void ReplPendingPartitionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_pending();
      runtime->free_repl_pending_partition_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplPendingPartitionOp::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // We know we are in a replicate context
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Perform the partitioning operation
      const ApEvent ready_event = thunk->perform_shard(this, runtime->forest,
        repl_ctx->owner_shard->shard_id, repl_ctx->shard_manager->total_shards);
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, 
          ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
      complete_operation(Runtime::protect_event(ready_event));
    }

    /////////////////////////////////////////////////////////////
    // Repl Dependent Partition Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplDependentPartitionOp(Runtime *rt)
      : DependentPartitionOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplDependentPartitionOp(
                                            const ReplDependentPartitionOp &rhs)
      : DependentPartitionOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::~ReplDependentPartitionOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp& ReplDependentPartitionOp::operator=(
                                            const ReplDependentPartitionOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_field(ReplicateContext *ctx, 
                                                       ShardID target,
                                                       ApEvent ready_event,
                                                       IndexPartition pid,
                                                       LogicalRegion handle, 
                                                       LogicalRegion parent,
                                                       FieldID fid,
                                                       MapperID id, 
                                                       MappingTagID t,
                                                       RtBarrier &deppart_bar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(pid, 
            handle.get_field_space(), fid, false/*range*/, 
            true/*use color space*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                      "of the color space elements for 'partition_by_field' "
                      "call in task %s (UID %lld)", fid, ctx->get_task_name(),
                      ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/); 
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement = 
        RegionRequirement(handle, LEGION_READ_ONLY, LEGION_EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ReplByFieldThunk(ctx, target, pid);
      mapping_barrier = deppart_bar;
      ctx->advance_replicate_barrier(deppart_bar, ctx->total_shards);
      partition_ready = ready_event;
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_image(ReplicateContext *ctx, 
#ifndef SHARD_BY_IMAGE
                                                       ShardID target,
#endif
                                                       ApEvent ready_event,
                                                       IndexPartition pid,
                                                   LogicalPartition projection,
                                             LogicalRegion parent, FieldID fid,
                                                   MapperID id, MappingTagID t,
                                                   ShardID shard, size_t total,
                                                        RtBarrier &deppart_bar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(pid, 
            projection.get_field_space(), fid, false/*range*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                      "of the destination index space elements for "
                      "'partition_by_image' call in task %s (UID %lld)",
                      fid, ctx->get_task_name(), ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      LogicalRegion proj_parent = 
        runtime->forest->get_parent_logical_region(projection);
      requirement = 
        RegionRequirement(proj_parent,LEGION_READ_ONLY,LEGION_EXCLUSIVE,parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
#ifdef SHARD_BY_IMAGE
      thunk = new ReplByImageThunk(ctx, pid, projection.get_index_partition(),
                                   shard, total);
#else
      thunk = new ReplByImageThunk(ctx, target, pid,
                                   projection.get_index_partition(),
                                   shard, total);
#endif
      mapping_barrier = deppart_bar;
      ctx->advance_replicate_barrier(deppart_bar, ctx->total_shards);
      partition_ready = ready_event;
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_image_range(
                                                         ReplicateContext *ctx, 
#ifndef SHARD_BY_IMAGE
                                                         ShardID target,
#endif
                                                         ApEvent ready_event,
                                                         IndexPartition pid,
                                                LogicalPartition projection,
                                                LogicalRegion parent,
                                                FieldID fid, MapperID id,
                                                MappingTagID t, ShardID shard, 
                                                size_t total_shards,
                                                RtBarrier &deppart_bar) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(pid, 
            projection.get_field_space(), fid, true/*range*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                      "of the destination index space elements for "
                      "'partition_by_image_range' call in task %s (UID %lld)",
                      fid, ctx->get_task_name(), ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      LogicalRegion proj_parent = 
        runtime->forest->get_parent_logical_region(projection);
      requirement = 
        RegionRequirement(proj_parent,LEGION_READ_ONLY,LEGION_EXCLUSIVE,parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
#ifdef SHARD_BY_IMAGE
      thunk = new ReplByImageRangeThunk(ctx, pid, 
                                        projection.get_index_partition(),
                                        shard, total_shards);
#else
      thunk = new ReplByImageRangeThunk(ctx, target, pid, 
                                        projection.get_index_partition(),
                                        shard, total_shards);
#endif
      mapping_barrier = deppart_bar;
      ctx->advance_replicate_barrier(deppart_bar, ctx->total_shards);
      partition_ready = ready_event;
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_preimage(ReplicateContext *ctx,
                                    ShardID target_shard, ApEvent ready_event,
                                    IndexPartition pid, IndexPartition proj,
                                    LogicalRegion handle, LogicalRegion parent,
                                    FieldID fid, MapperID id, MappingTagID t,
                                    RtBarrier &deppart_bar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(proj,
            handle.get_field_space(), fid, false/*range*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                      "of the range index space elements for "
                      "'partition_by_preimage' call in task %s (UID %lld)",
                      fid, ctx->get_task_name(), ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement = 
        RegionRequirement(handle, LEGION_READ_ONLY, LEGION_EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ReplByPreimageThunk(ctx, target_shard, pid, proj);
      mapping_barrier = deppart_bar;
      ctx->advance_replicate_barrier(deppart_bar, ctx->total_shards);
      partition_ready = ready_event;
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_preimage_range(
                                    ReplicateContext *ctx, ShardID target_shard,
                                    ApEvent ready_event,
                                    IndexPartition pid, IndexPartition proj,
                                    LogicalRegion handle, LogicalRegion parent,
                                    FieldID fid, MapperID id, MappingTagID t,
                                    RtBarrier &deppart_bar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(proj,
            handle.get_field_space(), fid, true/*range*/))
      {
        log_run.error("ERROR: Field size of field %d does not match the size "
                     "of the range index space elements for "
                     "'partition_by_preimage_range' call in task %s (UID %lld)",
                     fid, ctx->get_task_name(), ctx->get_unique_id());
        assert(false);
      }
#endif
      parent_task = ctx->get_task();
      initialize_operation(ctx, true/*track*/);
      // Start without the projection requirement, we'll ask
      // the mapper later if it wants to turn this into an index launch
      requirement = 
        RegionRequirement(handle, LEGION_READ_ONLY, LEGION_EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ReplByPreimageRangeThunk(ctx, target_shard, pid, proj);
      mapping_barrier = deppart_bar;
      ctx->advance_replicate_barrier(deppart_bar, ctx->total_shards);
      partition_ready = ready_event;
      if (runtime->legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_association(
                               ReplicateContext *ctx, LogicalRegion domain,
                               LogicalRegion domain_parent, FieldID fid,
                               IndexSpace range, MapperID id, MappingTagID tag,
                               RtBarrier &deppart_bar)
    //--------------------------------------------------------------------------
    {
      mapping_barrier = deppart_bar;
      ctx->advance_replicate_barrier(deppart_bar, ctx->total_shards);
      DependentPartitionOp::initialize_by_association(ctx, domain, 
                                domain_parent, fid, range, id, tag);
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_dependent_op();
      sharding_function = NULL;
#ifdef DEBUG_LEGION
      sharding_collective = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_dependent_op();
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      runtime->free_repl_dependent_partition_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::select_sharding_function(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
      assert(sharding_function == NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Do the mapper call to get the sharding function to use
      if (mapper == NULL)
        mapper = runtime->find_mapper(
            parent_ctx->get_executing_processor(), map_id);
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      output.chosen_functor = UINT_MAX;
      mapper->invoke_partition_select_sharding_functor(this, input, &output);
      if (output.chosen_functor == UINT_MAX)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s failed to pick a valid sharding functor for "
                      "dependent partition in task %s (UID %lld)", 
                      mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id())
      sharding_function = repl_ctx->shard_manager->find_sharding_function(
                                                    output.chosen_functor);
#ifdef DEBUG_LEGION
      assert(sharding_collective != NULL);
      sharding_collective->contribute(output.chosen_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(output.chosen_functor))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Mapper %s chose different sharding functions "
                      "for dependent partition op in task %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id())
#endif
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::select_partition_projection(void)
    //--------------------------------------------------------------------------
    {
      // TODO: put in a check here that all the shards pick the same partition
      DependentPartitionOp::select_partition_projection();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->check_privileges)
        check_privilege();
      // Before doing the dependence analysis we have to ask the
      // mapper whether it would like to make this an index space
      // operation or a single operation
      select_partition_projection();
      // Now that we know that we have the right region requirement we
      // can ask the mapper to also pick the sharding function
      select_sharding_function();
      // Do thise now that we've picked our region requirement
      initialize_privilege_path(privilege_path, requirement);
      if (runtime->legion_spy_enabled)
        log_requirement();
      ProjectionInfo projection_info;
      if (is_index_space)
        projection_info = ProjectionInfo(runtime, requirement, 
                                         launch_space, sharding_function);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   projection_info,
                                                   privilege_path,
                                                   map_applied_conditions);
      // Record this dependent partition op with the context so that it 
      // can track implicit dependences on it for later operations
      parent_ctx->update_current_implicit(this);
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      
      // Do different things if this is an index space point or a single point
      if (is_index_space)
      {
#ifdef DEBUG_LEGION
        assert(sharding_function != NULL);
#endif
        // Compute the local index space of points for this shard
        IndexSpace local_space =
          sharding_function->find_shard_space(repl_ctx->owner_shard->shard_id,
                                          launch_space, launch_space->handle);
        // If it's empty we're done, otherwise we go back on the queue
        if (!local_space.exists())
        {
#ifdef LEGION_SPY
          // Still have to do this for legion spy
          LegionSpy::log_operation_events(unique_op_id, 
              ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
          // We aren't participating directly, but we still have to 
          // participate in the collective operations
          const ApEvent done_event = 
            thunk->perform(this,runtime->forest,ApEvent::NO_AP_EVENT,instances);
          // We can try to early-complete this operation too
          request_early_complete(done_event);
          // We have no local points, so we can just trigger
          Runtime::phase_barrier_arrive(mapping_barrier, 1/*count*/);
          complete_mapping(mapping_barrier);
          complete_execution(Runtime::protect_event(done_event));
        }
        else // If we have valid points then we do the base call
        {
          if (remove_launch_space_reference(launch_space))
            delete launch_space;
          launch_space = runtime->forest->get_node(local_space);
          add_launch_space_reference(launch_space);
          DependentPartitionOp::trigger_ready();
        }
      }
      else
      {
        // Inform the thunk that we're eliding collectives since this
        // is a singular operation and not an index operation
        thunk->elide_collectives();
        // Shard 0 always owns dependent partition operations
        // If we own it we go on the queue, otherwise we complete early
        if (repl_ctx->owner_shard->shard_id != 0)
        {
#ifdef LEGION_SPY
          // Still have to do this for legion spy
          LegionSpy::log_operation_events(unique_op_id, 
              ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
          // We don't own it, so we can pretend like we
          // mapped and executed this task already
          Runtime::phase_barrier_arrive(mapping_barrier, 1/*count*/);
          complete_mapping(mapping_barrier);
          complete_execution();
        }
        else // If we're the shard then we do the base call
          DependentPartitionOp::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::finalize_mapping(void)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      if (!map_applied_conditions.empty())
        precondition = Runtime::merge_events(map_applied_conditions);
      Runtime::phase_barrier_arrive(mapping_barrier, 1/*count*/, precondition);
      if (!acquired_instances.empty())
        precondition = release_nonempty_acquired_instances(mapping_barrier, 
                                                           acquired_instances);
      else
        precondition = mapping_barrier;
      complete_mapping(precondition);
    }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplByFieldThunk::ReplByFieldThunk(
        ReplicateContext *ctx, ShardID target, IndexPartition p)
      : ByFieldThunk(p), 
        gather_collective(FieldDescriptorGather(ctx, target, COLLECTIVE_LOC_54))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ApEvent ReplDependentPartitionOp::ReplByFieldThunk::perform(
                              DependentPartitionOp *op,
                              RegionTreeForest *forest, ApEvent instances_ready,
                              const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      if (op->is_index_space)
      {
        gather_collective.contribute(instances_ready, instances);
        if (gather_collective.is_target())
        {
          ApEvent all_ready;
          const std::vector<FieldDataDescriptor> &full_descriptors =
            gather_collective.get_full_descriptors(all_ready);
          // Perform the operation
          ApEvent done = forest->create_partition_by_field(op, pid,
                                      full_descriptors, all_ready);
          gather_collective.notify_remote_complete(done);
          return done;
        }
        else // nothing else for us to do
          return gather_collective.get_complete_event();
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_field(op, pid, 
                                                 instances, instances_ready);
    }

    //--------------------------------------------------------------------------
#ifdef SHARD_BY_IMAGE
    ReplDependentPartitionOp::ReplByImageThunk::ReplByImageThunk(
                                          ReplicateContext *ctx, 
                                          IndexPartition p, IndexPartition proj,
                                          ShardID s, size_t total)
      : ByImageThunk(p, proj), 
        collective(FieldDescriptorExchange(ctx, COLLECTIVE_LOC_55)),
#else
    ReplDependentPartitionOp::ReplByImageThunk::ReplByImageThunk(
                                          ReplicateContext *ctx, ShardID target,
                                          IndexPartition p, IndexPartition proj,
                                          ShardID s, size_t total)
      : ByImageThunk(p, proj), 
        collective(FieldDescriptorGather(ctx, target, COLLECTIVE_LOC_55)),
#endif
        shard_id(s), total_shards(total)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ApEvent ReplDependentPartitionOp::ReplByImageThunk::perform(
                              DependentPartitionOp *op,
                              RegionTreeForest *forest, ApEvent instances_ready,
                              const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      if (op->is_index_space)
      {
#ifdef SHARD_BY_IMAGE
        // There is a special case here if we're projecting the same 
        // partition that we used to make the instances, if it is then
        // we can avoid needing to do the exchange at all
        if ((op->requirement.handle_type == PART_PROJECTION) &&
            (op->requirement.partition.get_index_partition() == projection))
        {
          // Make sure we elide our collective to avoid leaking anything
          collective.elide_collective();
          if (!instances.empty())
            return forest->create_partition_by_image_range(op, pid, projection,
                instances, instances_ready, shard_id, total_shards);
          else
            return ApEvent::NO_AP_EVENT;
        }
        // Do the all-to-all gather of the field data descriptors
        ApEvent all_ready = collective.exchange_descriptors(instances_ready,
                                                            instances);
        ApEvent done = forest->create_partition_by_image(op, pid, projection,
                  collective.descriptors, all_ready, shard_id, total_shards);
        return collective.exchange_completion(done);
#else
        collective.contribute(instances_ready, instances);
        if (collective.is_target())
        {
          ApEvent all_ready;
          const std::vector<FieldDataDescriptor> &full_descriptors =
            collective.get_full_descriptors(all_ready);
          // Perform the operation
          ApEvent done = forest->create_partition_by_image(op, pid,
                          projection, full_descriptors, all_ready);
          collective.notify_remote_complete(done);
          return done;
        }
        else // nothing else for us to do
          return collective.get_complete_event();
#endif
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_image(op, pid, projection, 
                                                 instances, instances_ready);
    }

    //--------------------------------------------------------------------------
#ifdef SHARD_BY_IMAGE
    ReplDependentPartitionOp::ReplByImageRangeThunk::ReplByImageRangeThunk(
                                          ReplicateContext *ctx, 
                                          IndexPartition p, IndexPartition proj,
                                          ShardID s, size_t total)
      : ByImageRangeThunk(p, proj), 
        collective(FieldDescriptorExchange(ctx, COLLECTIVE_LOC_60)),
#else
    ReplDependentPartitionOp::ReplByImageRangeThunk::ReplByImageRangeThunk(
                                          ReplicateContext *ctx, ShardID target,
                                          IndexPartition p, IndexPartition proj,
                                          ShardID s, size_t total)
      : ByImageRangeThunk(p, proj), 
        collective(FieldDescriptorGather(ctx, target, COLLECTIVE_LOC_60)),
#endif
        shard_id(s), total_shards(total)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ApEvent ReplDependentPartitionOp::ReplByImageRangeThunk::perform(
                              DependentPartitionOp *op,
                              RegionTreeForest *forest, ApEvent instances_ready,
                              const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      if (op->is_index_space)
      {
#ifdef SHARD_BY_IMAGE
        // There is a special case here if we're projecting the same 
        // partition that we used to make the instances, if it is then
        // we can avoid needing to do the exchange at all
        if ((op->requirement.handle_type == PART_PROJECTION) &&
            (op->requirement.partition.get_index_partition() == projection))
        {
          // Make sure we elide our collective to avoid leaking anything
          collective.elide_collective();
          if (!instances.empty())
            return forest->create_partition_by_image_range(op, pid, projection,
                instances, instances_ready, shard_id, total_shards);
          else
            return ApEvent::NO_AP_EVENT;
        }
        // Do the all-to-all gather of the field data descriptors
        ApEvent all_ready = collective.exchange_descriptors(instances_ready,
                                                            instances);
        ApEvent done = forest->create_partition_by_image_range(op, pid, 
            projection,collective.descriptors,all_ready,shard_id,total_shards);
        return collective.exchange_completion(done);   
#else
        collective.contribute(instances_ready, instances);
        if (collective.is_target())
        {
          ApEvent all_ready;
          const std::vector<FieldDataDescriptor> &full_descriptors =
            collective.get_full_descriptors(all_ready);
          // Perform the operation
          ApEvent done = forest->create_partition_by_image_range(op, pid,
                              projection, full_descriptors, all_ready);
          collective.notify_remote_complete(done);
          return done;
        }
        else // nothing else for us to do
          return collective.get_complete_event();
#endif
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_image_range(op, pid, projection, 
                                                 instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplByPreimageThunk::ReplByPreimageThunk(
                                          ReplicateContext *ctx, ShardID target,
                                          IndexPartition p, IndexPartition proj)
      : ByPreimageThunk(p, proj), 
        gather_collective(FieldDescriptorGather(ctx, target, COLLECTIVE_LOC_56))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ApEvent ReplDependentPartitionOp::ReplByPreimageThunk::perform(
                              DependentPartitionOp *op,
                              RegionTreeForest *forest, ApEvent instances_ready,
                              const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      if (op->is_index_space)
      {
        gather_collective.contribute(instances_ready, instances);
        if (gather_collective.is_target())
        {
          ApEvent all_ready;
          const std::vector<FieldDataDescriptor> &full_descriptors =
            gather_collective.get_full_descriptors(all_ready);
          // Perform the operation
          ApEvent done = forest->create_partition_by_preimage(op, pid, 
                              projection, full_descriptors, all_ready);
          gather_collective.notify_remote_complete(done);
          return done;
        }
        else // nothing else for us to do
          return gather_collective.get_complete_event();
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_preimage(op, pid, projection, 
                                                 instances, instances_ready);
    }
    
    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplByPreimageRangeThunk::
                 ReplByPreimageRangeThunk(ReplicateContext *ctx, ShardID target,
                                          IndexPartition p, IndexPartition proj)
      : ByPreimageRangeThunk(p, proj), 
        gather_collective(FieldDescriptorGather(ctx, target, COLLECTIVE_LOC_57))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ApEvent ReplDependentPartitionOp::ReplByPreimageRangeThunk::perform(
                              DependentPartitionOp *op,
                              RegionTreeForest *forest, ApEvent instances_ready,
                              const std::vector<FieldDataDescriptor> &instances)
    //--------------------------------------------------------------------------
    {
      if (op->is_index_space)
      {
        gather_collective.contribute(instances_ready, instances);
        if (gather_collective.is_target())
        {
          ApEvent all_ready;
          const std::vector<FieldDataDescriptor> &full_descriptors =
            gather_collective.get_full_descriptors(all_ready);
          // Perform the operation
          ApEvent done = forest->create_partition_by_preimage_range(op, pid, 
                                    projection, full_descriptors, all_ready);
          gather_collective.notify_remote_complete(done);
          return done;
        }
        else // nothing else for us to do
          return gather_collective.get_complete_event();
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_preimage_range(op, pid, projection, 
                                                 instances, instances_ready);
    }

    /////////////////////////////////////////////////////////////
    // Repl Must Epoch Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplMustEpochOp::ReplMustEpochOp(Runtime *rt)
      : MustEpochOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplMustEpochOp::ReplMustEpochOp(const ReplMustEpochOp &rhs)
      : MustEpochOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplMustEpochOp::~ReplMustEpochOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplMustEpochOp& ReplMustEpochOp::operator=(const ReplMustEpochOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_must_epoch_op();
      sharding_functor = UINT_MAX;
      sharding_function = NULL;
      mapping_collective_id = 0;
      collective_map_must_epoch_call = false;
      mapping_broadcast = NULL;
      mapping_exchange = NULL;
      dependence_exchange = NULL;
      completion_exchange = NULL;
#ifdef DEBUG_LEGION
      sharding_collective = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_must_epoch_op(); 
      shard_single_tasks.clear();
      runtime->free_repl_epoch_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::instantiate_tasks(InnerContext *ctx, 
                                            const MustEpochLauncher &launcher)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(ctx);
#endif
      // Initialize operations for everything in the launcher
      // Note that we do not track these operations as we want them all to
      // appear as a single operation to the parent context in order to
      // avoid deadlock with the maximum window size.
      indiv_tasks.resize(launcher.single_tasks.size());
      for (unsigned idx = 0; idx < launcher.single_tasks.size(); idx++)
      {
        ReplIndividualTask *task = 
          runtime->get_available_repl_individual_task();
        task->initialize_task(ctx, launcher.single_tasks[idx], false/*track*/);
        task->set_must_epoch(this, idx, true/*register*/);
        // If we have a trace, set it for this operation as well
        if (trace != NULL)
          task->set_trace(trace, NULL);
        task->must_epoch_task = true;
        task->initialize_replication(repl_ctx);
        task->index_domain = this->launch_domain;
        task->sharding_space = launcher.sharding_space;
#ifdef DEBUG_LEGION
        task->set_sharding_collective(new ShardingGatherCollective(repl_ctx,
                                      0/*owner shard*/, COLLECTIVE_LOC_59));
#endif
        indiv_tasks[idx] = task;
      }
      indiv_triggered.resize(indiv_tasks.size(), false);
      index_tasks.resize(launcher.index_tasks.size());
      for (unsigned idx = 0; idx < launcher.index_tasks.size(); idx++)
      {
        IndexSpace launch_space = launcher.index_tasks[idx].launch_space;
        if (!launch_space.exists())
          launch_space = ctx->find_index_launch_space(
                          launcher.index_tasks[idx].launch_domain);
        ReplIndexTask *task = runtime->get_available_repl_index_task();
        task->initialize_task(ctx, launcher.index_tasks[idx],
                              launch_space, false/*track*/);
        task->set_must_epoch(this, indiv_tasks.size()+idx, 
                                         true/*register*/);
        if (trace != NULL)
          task->set_trace(trace, NULL);
        task->must_epoch_task = true;
        task->initialize_replication(repl_ctx);
        task->sharding_space = launcher.sharding_space;
#ifdef DEBUG_LEGION
        task->set_sharding_collective(new ShardingGatherCollective(repl_ctx,
                                      0/*owner shard*/, COLLECTIVE_LOC_59));
#endif
        index_tasks[idx] = task;
      }
      index_triggered.resize(index_tasks.size(), false);
    }

    //--------------------------------------------------------------------------
    FutureMapImpl* ReplMustEpochOp::create_future_map(TaskContext *ctx,
              const Domain &domain, IndexSpace shard_space, RtUserEvent deleted)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(ctx);
#endif
      Domain shard_domain;
      if (shard_space.exists())
        runtime->forest->find_launch_space_domain(shard_space, shard_domain);
      else
        shard_domain = domain;
      return new ReplFutureMapImpl(repl_ctx, this, 
          Runtime::protect_event(get_completion_event()), domain, shard_domain,
          runtime, runtime->get_available_distributed_id(), 
          runtime->address_space);
    }

    //--------------------------------------------------------------------------
    MapperManager* ReplMustEpochOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
      Processor mapper_proc = parent_ctx->get_executing_processor();
      MapperManager *mapper = runtime->find_mapper(mapper_proc, map_id);
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // We want to do the map must epoch call
      // First find all the tasks that we own on this shard
      Domain shard_domain = launch_domain;
      if (sharding_space.exists())
        runtime->forest->find_launch_space_domain(sharding_space, shard_domain);
      for (std::vector<SingleTask*>::const_iterator it = 
            single_tasks.begin(); it != single_tasks.end(); it++)
      {
        const ShardID shard = 
          sharding_function->find_owner((*it)->index_point, shard_domain);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_owner_shard((*it)->get_unique_id(), shard);
        // If it is not our shard then we don't own it
        if (shard != repl_ctx->owner_shard->shard_id)
          continue;
        shard_single_tasks.insert(*it);
      }
      // Find the set of constraints that apply to our local set of tasks
      std::vector<Mapper::MappingConstraint> local_constraints;
      std::vector<unsigned> original_constraint_indexes;
      for (unsigned idx = 0; idx < input.constraints.size(); idx++)
      {
        bool is_local = false;
        for (std::vector<const Task*>::const_iterator it = 
              input.constraints[idx].constrained_tasks.begin(); it !=
              input.constraints[idx].constrained_tasks.end(); it++)
        {
          SingleTask *single = static_cast<SingleTask*>(const_cast<Task*>(*it));
          if (shard_single_tasks.find(single) == shard_single_tasks.end())
            continue;
          is_local = true;
          break;
        }
        if (is_local)
        {
          local_constraints.push_back(input.constraints[idx]);
          original_constraint_indexes.push_back(idx);
        }
      }
      if (collective_map_must_epoch_call)
      {
        // Update the input tasks for our subset
        std::vector<const Task*> all_tasks(shard_single_tasks.begin(),
                                           shard_single_tasks.end());
        input.tasks.swap(all_tasks);
        // Sort them again by their index points to for determinism
        std::sort(input.tasks.begin(), input.tasks.end(), single_task_sorter);
        // Update the constraints to contain just our subset
        const size_t total_constraints = input.constraints.size();
        input.constraints.swap(local_constraints);
        // Fill in our shard mapping and local shard info
        input.shard_mapping = repl_ctx->shard_manager->shard_mapping;
        input.local_shard = repl_ctx->owner_shard->shard_id;
        // Update the outputs
        output.task_processors.resize(input.tasks.size());
        output.constraint_mappings.resize(input.constraints.size());
        output.weights.resize(input.constraints.size());
        // Now we can do the mapper call
        mapper->invoke_map_must_epoch(this, &input, &output);
        // Now we need to exchange our mapping decisions between all the shards
#ifdef DEBUG_LEGION
        assert(mapping_exchange == NULL);
        assert(mapping_collective_id > 0);
#endif
        mapping_exchange = 
          new MustEpochMappingExchange(repl_ctx, mapping_collective_id);
        mapping_exchange->exchange_must_epoch_mappings(
                  repl_ctx->owner_shard->shard_id,
                  repl_ctx->shard_manager->total_shards, total_constraints,
                  input.tasks, all_tasks, output.task_processors,
                  original_constraint_indexes, output.constraint_mappings,
                  output.weights, *get_acquired_instances_ref());
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(mapping_broadcast == NULL);
        assert(mapping_collective_id > 0);
#endif
        mapping_broadcast = new MustEpochMappingBroadcast(repl_ctx, 
                                  0/*owner shard*/, mapping_collective_id);
        // Do the mapper call on shard 0 and then broadcast the results
        if (repl_ctx->owner_shard->shard_id == 0)
        {
          mapper->invoke_map_must_epoch(this, &input, &output);
          mapping_broadcast->broadcast(output.task_processors,
                                       output.constraint_mappings);
        }
        else
          mapping_broadcast->receive_results(output.task_processors,
              original_constraint_indexes, output.constraint_mappings,
              *get_acquired_instances_ref());
      }
      // No need to do any checks, the base class handles that
      return mapper;
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::map_and_distribute(std::set<RtEvent> &tasks_mapped,
                                             std::set<ApEvent> &tasks_complete)
    //--------------------------------------------------------------------------
    {
      // Perform the mapping
      map_replicate_tasks();
      mapping_dependences.clear();
      // We have to exchange mapping and completion events with all the
      // other shards as well
      std::set<RtEvent> local_tasks_mapped;
      std::set<ApEvent> local_tasks_complete;
      for (std::vector<IndividualTask*>::const_iterator it = 
            indiv_tasks.begin(); it != indiv_tasks.end(); it++)
      {
        local_tasks_mapped.insert((*it)->get_mapped_event());
        local_tasks_complete.insert((*it)->get_completion_event());
      }
      for (std::vector<IndexTask*>::const_iterator it = 
            index_tasks.begin(); it != index_tasks.end(); it++)
      {
        local_tasks_mapped.insert((*it)->get_mapped_event());
        local_tasks_complete.insert((*it)->get_completion_event());
      }
      RtEvent local_mapped = Runtime::merge_events(local_tasks_mapped);
      tasks_mapped.insert(local_mapped);
      ApEvent local_complete = Runtime::merge_events(NULL,local_tasks_complete);
      tasks_complete.insert(local_complete);
#ifdef DEBUG_LEGION
      assert(completion_exchange != NULL);
#endif
      completion_exchange->exchange_must_epoch_completion(
          local_mapped, local_complete, tasks_mapped, tasks_complete);
      // Then we can distribute the tasks
      distribute_replicate_tasks();
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      Processor mapper_proc = parent_ctx->get_executing_processor();
      MapperManager *mapper = runtime->find_mapper(mapper_proc, map_id);
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Select our sharding functor and then do the base call
      this->individual_tasks.resize(indiv_tasks.size());
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        this->individual_tasks[idx] = indiv_tasks[idx];
      this->index_space_tasks.resize(index_tasks.size());
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        this->index_space_tasks[idx] = index_tasks[idx];
      Mapper::SelectShardingFunctorInput sharding_input;
      sharding_input.shard_mapping = repl_ctx->shard_manager->shard_mapping;
      Mapper::MustEpochShardingFunctorOutput sharding_output;
      sharding_output.chosen_functor = UINT_MAX;
      sharding_output.collective_map_must_epoch_call = false;
      mapper->invoke_must_epoch_select_sharding_functor(this,
                                    &sharding_input, &sharding_output);
      // We can clear these now that we don't need them anymore
      individual_tasks.clear();
      index_space_tasks.clear();
      // Check that we have a sharding ID
      if (sharding_output.chosen_functor == UINT_MAX)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
            "Invalid mapper output from invocation of "
            "'map_must_epoch' on mapper %s. Mapper failed to specify "
            "a valid sharding ID for a must epoch operation in control "
            "replicated context of task %s (UID %lld).",
            mapper->get_mapper_name(), repl_ctx->get_task_name(),
            repl_ctx->get_unique_id())
      this->sharding_functor = sharding_output.chosen_functor;
      this->collective_map_must_epoch_call = 
        sharding_output.collective_map_must_epoch_call;
#ifdef DEBUG_LEGION
      assert(sharding_function == NULL);
      // Check that the sharding IDs are all the same
      assert(sharding_collective != NULL);
      // Contribute the result
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() && 
          !sharding_collective->validate(this->sharding_functor))
      {
        log_run.error("ERROR: Mapper %s chose different sharding functions "
                      "for must epoch launch in %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id());
        assert(false); 
      }
      ReplFutureMapImpl *impl = 
          dynamic_cast<ReplFutureMapImpl*>(result_map.impl);
      assert(impl != NULL);
#else
      ReplFutureMapImpl *impl = 
          static_cast<ReplFutureMapImpl*>(result_map.impl);
#endif
      // Set the future map sharding functor
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      impl->set_sharding_function(sharding_function);
      // Set the sharding functor for all the point and index tasks too
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        ReplIndividualTask *task = 
          static_cast<ReplIndividualTask*>(indiv_tasks[idx]);
        task->set_sharding_function(sharding_functor, sharding_function);
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        ReplIndexTask *task = static_cast<ReplIndexTask*>(index_tasks[idx]);
        task->set_sharding_function(sharding_functor, sharding_function);
      }
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      // We have to delete these here to make sure that they are
      // unregistered with the context before the context is deleted
      if (mapping_broadcast != NULL)
        delete mapping_broadcast;
      if (mapping_exchange != NULL)
        delete mapping_exchange;
      if (dependence_exchange != NULL)
        delete dependence_exchange;
      if (completion_exchange != NULL)
        delete completion_exchange;
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      MustEpochOp::trigger_commit();
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::map_replicate_tasks(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(dependence_exchange != NULL);
      assert(single_tasks.size() == mapping_dependences.size());
#endif
      std::map<DomainPoint,RtUserEvent> mapped_events;
      for (std::set<SingleTask*>::const_iterator it = 
            shard_single_tasks.begin(); it != shard_single_tasks.end(); it++)
        mapped_events[(*it)->index_point] = Runtime::create_rt_user_event();
      // Now exchange completion events for the point tasks we own
      // and end up with a set of the completion event for each task
      // First compute the set of mapped events for the points that we own
      dependence_exchange->exchange_must_epoch_dependences(mapped_events);

      MustEpochMapArgs args(const_cast<ReplMustEpochOp*>(this));
      std::set<RtEvent> local_mapped_events;
      // For correctness we still have to abide by the mapping dependences
      // computed on the individual tasks while we are mapping them
      for (unsigned idx = 0; idx < single_tasks.size(); idx++)
      {
        // Check to see if it is one of the ones that we own
        if (shard_single_tasks.find(single_tasks[idx]) == 
            shard_single_tasks.end())
        {
          // We don't own this point
          // We still need to do some work for individual tasks
          // to exchange versioning information, but no such 
          // work is necessary for point tasks
          SingleTask *task = single_tasks[idx];
          task->shard_off(mapped_events[task->index_point]);
          continue;
        }
        // Figure out our preconditions
        std::set<RtEvent> preconditions;
        for (std::set<unsigned>::const_iterator it = 
              mapping_dependences[idx].begin(); it != 
              mapping_dependences[idx].end(); it++)
        {
#ifdef DEBUG_LEGION
          assert((*it) < idx);
#endif
          preconditions.insert(mapped_events[single_tasks[*it]->index_point]);
        }
        args.task = single_tasks[idx];
        RtEvent done;
        if (!preconditions.empty())
        {
          RtEvent precondition = Runtime::merge_events(preconditions);
          done = runtime->issue_runtime_meta_task(args, 
                LG_THROUGHPUT_DEFERRED_PRIORITY, precondition); 
        }
        else
          done = runtime->issue_runtime_meta_task(args, 
                      LG_THROUGHPUT_DEFERRED_PRIORITY);
        local_mapped_events.insert(done);
        // We can trigger our completion event once the task is done
        RtUserEvent mapped = mapped_events[single_tasks[idx]->index_point];
        Runtime::trigger_event(mapped, done);
      }
      // Now we have to wait for all our mapping operations to be done
      if (!local_mapped_events.empty())
      {
        RtEvent mapped_event = Runtime::merge_events(local_mapped_events);
        mapped_event.wait();
      }
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::distribute_replicate_tasks(void) const
    //--------------------------------------------------------------------------
    {
      // We only want to distribute the points that are owned by our shard
      ReplMustEpochOp *owner = const_cast<ReplMustEpochOp*>(this);
      MustEpochDistributorArgs dist_args(owner);
      MustEpochLauncherArgs launch_args(owner);
      std::set<RtEvent> wait_events;
      for (std::vector<IndividualTask*>::const_iterator it = 
            indiv_tasks.begin(); it != indiv_tasks.end(); it++)
      {
        // Skip any points that we do not own on this shard
        if (shard_single_tasks.find(*it) == shard_single_tasks.end())
          continue;
        if (!runtime->is_local((*it)->target_proc))
        {
          dist_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(dist_args, 
                LG_THROUGHPUT_DEFERRED_PRIORITY);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(launch_args,
                  LG_THROUGHPUT_DEFERRED_PRIORITY);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      for (std::set<SliceTask*>::const_iterator it = 
            slice_tasks.begin(); it != slice_tasks.end(); it++)
      {
        // Check to see if we either do or not own this slice
        // We currently do not support mixed slices for which
        // we only own some of the points
        bool contains_any = false;
        bool contains_all = true;
        for (std::vector<PointTask*>::const_iterator pit = 
              (*it)->points.begin(); pit != (*it)->points.end(); pit++)
        {
          if (shard_single_tasks.find(*pit) != shard_single_tasks.end())
            contains_any = true;
          else if (contains_all)
          {
            contains_all = false;
            if (contains_any) // At this point we have all the answers
              break;
          }
        }
        if (!contains_any)
          continue;
        if (!contains_all)
        {
          Processor mapper_proc = parent_ctx->get_executing_processor();
          MapperManager *mapper = runtime->find_mapper(mapper_proc, map_id);
          REPORT_LEGION_FATAL(ERROR_INVALID_MAPPER_OUTPUT,
                              "Mapper %s specified a slice for a must epoch "
                              "launch in control replicated task %s "
                              "(UID %lld) for which not all the points "
                              "mapped to the same shard. Legion does not "
                              "currently support this use case. Please "
                              "specify slices and a sharding function to "
                              "ensure that all the points in a slice are "
                              "owned by the same shard", 
                              mapper->get_mapper_name(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id())
        }
        (*it)->update_target_processor();
        if (!runtime->is_local((*it)->target_proc))
        {
          dist_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(dist_args, 
                LG_THROUGHPUT_DEFERRED_PRIORITY);
          if (wait.exists())
            wait_events.insert(wait);
        }
        else
        {
          launch_args.task = *it;
          RtEvent wait = 
            runtime->issue_runtime_meta_task(launch_args,
                 LG_THROUGHPUT_DEFERRED_PRIORITY);
          if (wait.exists())
            wait_events.insert(wait);
        }
      }
      if (!wait_events.empty())
      {
        RtEvent dist_event = Runtime::merge_events(wait_events);
        dist_event.wait();
      }
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping_collective_id == 0);
      assert(mapping_broadcast == NULL);
      assert(mapping_exchange == NULL);
      assert(dependence_exchange == NULL);
      assert(completion_exchange == NULL);
#endif
      // We can't actually make a collective for the mapping yet because we 
      // don't know if we are going to broadcast or exchange so we just get
      // a collective ID that we will use later 
      mapping_collective_id = ctx->get_next_collective_index(COLLECTIVE_LOC_58);
      dependence_exchange = 
        new MustEpochDependenceExchange(ctx, COLLECTIVE_LOC_70);
      completion_exchange = 
        new MustEpochCompletionExchange(ctx, COLLECTIVE_LOC_73);
    }

    //--------------------------------------------------------------------------
    Domain ReplMustEpochOp::get_shard_domain(void) const
    //--------------------------------------------------------------------------
    {
      if (sharding_space.exists())
      {
        Domain shard_domain;
        runtime->forest->find_launch_space_domain(sharding_space, shard_domain);
        return shard_domain;
      }
      else
        return launch_domain;
    }

    /////////////////////////////////////////////////////////////
    // Repl Timing Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTimingOp::ReplTimingOp(Runtime *rt)
      : TimingOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTimingOp::ReplTimingOp(const ReplTimingOp &rhs)
      : TimingOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplTimingOp::~ReplTimingOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTimingOp& ReplTimingOp::operator=(const ReplTimingOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplTimingOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_timing();
      timing_collective = NULL;
    }

    //--------------------------------------------------------------------------
    void ReplTimingOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      if (timing_collective != NULL)
      {
        delete timing_collective;
        timing_collective = NULL;
      }
      deactivate_timing();
      runtime->free_repl_timing_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplTimingOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Shard 0 will handle the timing operation so do the normal mapping
      if (repl_ctx->owner_shard->shard_id > 0)
      {
        complete_mapping();
        RtEvent result_ready = 
          timing_collective->perform_collective_wait(false/*block*/);
        if (result_ready.exists() && !result_ready.has_triggered())
        {
          // Defer completion until the value is ready
          DeferredExecuteArgs deferred_execute_args(this);
          runtime->issue_runtime_meta_task(deferred_execute_args,
                  LG_THROUGHPUT_DEFERRED_PRIORITY, result_ready);
        }
        else
          deferred_execute();
      }
      else // Shard 0 does the normal timing operation
        TimingOp::trigger_mapping();
    } 

    //--------------------------------------------------------------------------
    void ReplTimingOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Shard 0 will handle the timing operation
      if (repl_ctx->owner_shard->shard_id > 0)     
      {
        long long value = timing_collective->get_value(false/*already waited*/);
        result.impl->set_result(&value, sizeof(value), false);
      }
      else
      {
        // Perform the measurement and then arrive on the barrier
        // with the result to broadcast it to the other shards
        switch (measurement)
        {
          case LEGION_MEASURE_SECONDS:
            {
              double value = Realm::Clock::current_time();
              result.impl->set_result(&value, sizeof(value), false);
              long long *ptr = reinterpret_cast<long long*>(&value);
              timing_collective->broadcast(*ptr);
              break;
            }
          case LEGION_MEASURE_MICRO_SECONDS:
            {
              long long value = Realm::Clock::current_time_in_microseconds();
              result.impl->set_result(&value, sizeof(value), false);
              timing_collective->broadcast(value);
              break;
            }
          case LEGION_MEASURE_NANO_SECONDS:
            {
              long long value = Realm::Clock::current_time_in_nanoseconds();
              result.impl->set_result(&value, sizeof(value), false);
              timing_collective->broadcast(value);
              break;
            }
          default:
            assert(false); // should never get here
        }
      }
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Repl All Reduce Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplAllReduceOp::ReplAllReduceOp(Runtime *rt)
      : AllReduceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplAllReduceOp::ReplAllReduceOp(const ReplAllReduceOp &rhs)
      : AllReduceOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplAllReduceOp::~ReplAllReduceOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplAllReduceOp& ReplAllReduceOp::operator=(const ReplAllReduceOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(redop != NULL);
      assert(exchange_collective == NULL);
      assert(all_reduce_collective == NULL);
#endif
      if (deterministic)
        exchange_collective = 
          new FutureExchange(ctx, redop->sizeof_rhs, COLLECTIVE_LOC_97);
      else
        all_reduce_collective = 
          new AllReduceOpCollective(COLLECTIVE_LOC_97, ctx, redop);
    }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_all_reduce();
      result_buffer = NULL;
      exchange_collective = NULL;
      all_reduce_collective = NULL;
    }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_all_reduce();
      if (exchange_collective != NULL)
        delete exchange_collective;
      if (all_reduce_collective != NULL)
        delete all_reduce_collective;
      runtime->free_repl_all_reduce_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::deferred_execute(void)
    //--------------------------------------------------------------------------
    {
      // See if this is our first pass through to perform the reduction
      if (result_buffer == NULL)
      { 
        // First perform the reduction on our shard local futures
        std::map<DomainPoint,FutureImpl*> futures;
        future_map.impl->get_shard_local_futures(futures);
        result_buffer = malloc(redop->sizeof_rhs);
        redop->init(result_buffer, 1/*count*/);
        for (std::map<DomainPoint,FutureImpl*>::const_iterator it = 
              futures.begin(); it != futures.end(); it++)
        {
          FutureImpl *impl = it->second;
          const size_t future_size = impl->get_untyped_size(true/*internal*/);
          if (future_size != redop->sizeof_rhs)
            REPORT_LEGION_ERROR(ERROR_FUTURE_MAP_REDOP_TYPE_MISMATCH,
                "Future in future map reduction in task %s (UID %lld) does not "
                "have the right input size for the given reduction operator. "
                "Future has size %zd bytes but reduction operator expects "
                "RHS inputs of %zd bytes.", parent_ctx->get_task_name(),
                parent_ctx->get_unique_id(), future_size, redop->sizeof_rhs)
          const void *data = 
                        impl->get_untyped_result(true,NULL,true/*internal*/);
          redop->fold(result_buffer, data, 1/*count*/, true/*exclusive*/);
        }
        if (runtime->legion_spy_enabled)
        {
          for (std::map<DomainPoint,FutureImpl*>::const_iterator it = 
                futures.begin(); it != futures.end(); it++)
          {
            FutureImpl *impl = it->second;
            const ApEvent ready_event = impl->get_ready_event();
            if (ready_event.exists())
              LegionSpy::log_future_use(unique_op_id, ready_event);
          }
        }
        // Now do the exchange across the shards
        RtEvent defer;
        if (deterministic)
          defer = exchange_collective->exchange_futures(result_buffer);
        else
          defer = all_reduce_collective->async_reduce(result_buffer);
        if (defer.exists() && !defer.has_triggered())
        {
          DeferredExecuteArgs args(this);
          runtime->issue_runtime_meta_task(args, 
              LG_THROUGHPUT_DEFERRED_PRIORITY, defer);
          return;
        }
      }
      // If we make it here then we can get the results of the
      // reductions across the shards
      if (deterministic)
        exchange_collective->reduce_futures(redop, result_buffer);
      else
        all_reduce_collective->sync_result(result_buffer);
      // Tell the future about the final result which it will own
      result.impl->set_result(result_buffer, redop->sizeof_rhs, true/*own*/);
#ifdef LEGION_SPY
      // Still have to do this call to let Legion Spy know we're done
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      ApEvent::NO_AP_EVENT);
#endif
      // Mark that we are done executing which will complete the future
      // as soon as this operation is complete
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Repl Fence Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplFenceOp::ReplFenceOp(Runtime *rt)
      : FenceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplFenceOp::ReplFenceOp(const ReplFenceOp &rhs)
      : FenceOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplFenceOp::~ReplFenceOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplFenceOp& ReplFenceOp::operator=(const ReplFenceOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      FenceOp::activate();
      mapping_fence_barrier = RtBarrier::NO_RT_BARRIER;
      execution_fence_barrier = ApBarrier::NO_AP_BARRIER;
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fence();
      runtime->free_repl_fence_op(this);
    }

    //--------------------------------------------------------------------------
    Future ReplFenceOp::initialize_repl_fence(ReplicateContext *ctx, 
                                      FenceKind k, bool need_future, bool track)
    //--------------------------------------------------------------------------
    {
      Future f = initialize(ctx, k, need_future, track);
      mapping_fence_barrier = ctx->get_next_mapping_fence_barrier();
      if (fence_kind == EXECUTION_FENCE)
        execution_fence_barrier = ctx->get_next_execution_fence_barrier();
      return f;
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      switch (fence_kind)
      {
        case MAPPING_FENCE:
          {
            // Do our arrival
            if (!map_applied_conditions.empty())
              Runtime::phase_barrier_arrive(mapping_fence_barrier, 1/*count*/,
                  Runtime::merge_events(map_applied_conditions));
            else
              Runtime::phase_barrier_arrive(mapping_fence_barrier, 1/*count*/);
            // We're mapped when everyone is mapped
            complete_mapping(mapping_fence_barrier);
            complete_execution();
            break;
          }
        case EXECUTION_FENCE:
          {
            // If we're recording find all the prior event dependences
            if (is_recording())
              tpl->find_execution_fence_preconditions(execution_preconditions);
            const PhysicalTraceInfo trace_info(this, 0/*index*/, true/*init*/);
            // Do our arrival on our mapping fence, we're mapped when
            // everyone is mapped
            if (!map_applied_conditions.empty())
              Runtime::phase_barrier_arrive(mapping_fence_barrier, 1/*count*/,
                  Runtime::merge_events(map_applied_conditions));
            else
              Runtime::phase_barrier_arrive(mapping_fence_barrier, 1/*count*/);
            complete_mapping(mapping_fence_barrier);
            // We arrive on our barrier when all our previous operations
            // have finished executing
            ApEvent execution_fence_precondition;
            if (!execution_preconditions.empty())
              execution_fence_precondition = 
                  Runtime::merge_events(&trace_info, execution_preconditions);
            if (is_recording())
              trace_info.record_complete_replay(this, 
                        execution_fence_precondition);
            Runtime::phase_barrier_arrive(execution_fence_barrier, 1/*count*/, 
                                          execution_fence_precondition);
            // We can always trigger the completion event when these are done
            request_early_complete(execution_fence_barrier);
            if (!execution_fence_barrier.has_triggered())
            {
              RtEvent wait_on = Runtime::protect_event(execution_fence_barrier);
              complete_execution(wait_on);
            }
            else
              complete_execution();
            break;
          }
        default:
          assert(false); // should never get here
      }
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      // free up these barriers since we didn't use them
      Runtime::phase_barrier_arrive(mapping_fence_barrier, 1/*count*/);
      FenceOp::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplFenceOp::complete_replay(ApEvent complete_event)
    //--------------------------------------------------------------------------
    {
      Runtime::phase_barrier_arrive(execution_fence_barrier, 
                                    1/*count*/, complete_event);
      FenceOp::complete_replay(execution_fence_barrier);
    }

    /////////////////////////////////////////////////////////////
    // Repl Map Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplMapOp::ReplMapOp(Runtime *rt)
      : MapOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplMapOp::ReplMapOp(const ReplMapOp &rhs)
      : MapOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplMapOp::~ReplMapOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplMapOp& ReplMapOp::operator=(const ReplMapOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplMapOp::initialize_replication(ReplicateContext *ctx,RtBarrier &bar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(exchange == NULL);
      assert(view_did_broadcast == NULL);
      assert(sharded_view == NULL);
#endif
      inline_barrier = bar;
      ctx->advance_replicate_barrier(bar, ctx->total_shards);
      // We only check the results of the mapping if the runtime requests it
      // We can skip the check though if this is a read-only requirement
      if (!IS_READ_ONLY(requirement))
        exchange = new ShardedMappingExchange(COLLECTIVE_LOC_74, ctx,
                           ctx->owner_shard->shard_id, !runtime->unsafe_mapper);
      if (IS_WRITE(requirement))
      {
        // We need a second generation of the barrier for writes
        ctx->advance_replicate_barrier(bar, ctx->total_shards);
        // We need a third generation of the barrirer if we're not discarding
        // the previous version of the barrier so we can make sure all the
        // updates have been performed before we register our users
        if (!IS_DISCARD(requirement))
          ctx->advance_replicate_barrier(bar, ctx->total_shards);
        view_did_broadcast = 
          new ValueBroadcast<DistributedID>(ctx, 0/*owner*/, COLLECTIVE_LOC_75);
        // if we're shard 0 then get the distributed id and send it out
        if (ctx->owner_shard->shard_id == 0)
        {
          DistributedID view_did = runtime->get_available_distributed_id();
          // make it and register it with the runtime
          sharded_view = new ShardedView(runtime->forest,
            view_did, runtime->address_space, true/*register now*/);
          // then broadcast the result out so the other nodes can grab it
          view_did_broadcast->broadcast(sharded_view->did);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplMapOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(inline_barrier.exists());
#endif
      // Compute the version numbers for this mapping operation
      std::set<RtEvent> preconditions;
      runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                   requirement, 
                                                   version_info,
                                                   preconditions);
      if ((view_did_broadcast != NULL) && (sharded_view == NULL))
      {
        // Get the distributed ID for the sharded view and request it
        const DistributedID sharded_view_did = view_did_broadcast->get_value();
        RtEvent ready;
        sharded_view = static_cast<ShardedView*>(
            runtime->find_or_request_logical_view(sharded_view_did, ready));
        if (ready.exists())
          preconditions.insert(ready);
      }
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void ReplMapOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      const PhysicalTraceInfo trace_info(this, 0/*index*/, true/*init*/);
      // If we have any wait preconditions from phase barriers or 
      // grants then we use them to compute a precondition for doing
      // any copies or anything else for this operation
      ApEvent init_precondition = execution_fence_event;
      if (!wait_barriers.empty() || !grants.empty())
      {
        ApEvent sync_precondition = 
          merge_sync_preconditions(trace_info, grants, wait_barriers);
        if (sync_precondition.exists())
        {
          if (init_precondition.exists())
            init_precondition = Runtime::merge_events(&trace_info, 
                                  init_precondition, sync_precondition); 
          else
            init_precondition = sync_precondition;
        }
      }
      InstanceSet mapped_instances;
      // If we are remapping then we know the answer
      // so we don't need to do any premapping
      bool record_valid = true;
      if (remap_region)
        region.impl->get_references(mapped_instances);
      else
        record_valid = invoke_mapper(mapped_instances);
      // First kick off the exchange to get that in flight
      std::vector<InstanceView*> mapped_views;
      {
        InnerContext *context = find_physical_context(0/*index*/, requirement);
        context->convert_target_views(mapped_instances, mapped_views);
        if (exchange != NULL)
          exchange->initiate_exchange(mapped_instances, mapped_views);
      }
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      const bool is_owner_shard = (repl_ctx->owner_shard->shard_id == 0); 
      ApEvent effects_done;
      // What we do next depends on the privileges
      if (IS_REDUCE(requirement))
      {
        // Shard 0 updates the equivalence sets with its reduction buffer
        // Everyone else just needs to do their registration
        if (!is_owner_shard)
        {
          InnerContext *context = find_physical_context(0/*index*/,requirement);
          context->convert_target_views(mapped_instances, mapped_views); 
          RegionNode *node = runtime->forest->get_node(requirement.region);
          UpdateAnalysis *analysis = new UpdateAnalysis(runtime, this, 
                                      0/*index*/, version_info,
                                      requirement, node, mapped_instances,
                                      mapped_views,trace_info,init_precondition,
                                      termination_event, true/*track effects*/,
                                      false/*check initialized*/, record_valid,
                                      false/*skip output*/);
          analysis->add_reference();
          // Note that this call will clean up the analysis allocation
          effects_done = runtime->forest->physical_perform_registration(
                analysis, mapped_instances, trace_info, map_applied_conditions);
        }
        else
          effects_done = 
            runtime->forest->physical_perform_updates_and_registration(
                requirement, version_info, this, 0/*index*/, init_precondition,
                termination_event, mapped_instances, trace_info, 
                map_applied_conditions,
#ifdef DEBUG_LEGION
                get_logging_name(), unique_op_id,
#endif
                true/*track effects*/);
        // Complete the exchange
        exchange->complete_exchange(this, sharded_view, 
                                    mapped_instances, map_applied_conditions);
      }
      else if (IS_WRITE(requirement) && IS_DISCARD(requirement))
      {
#ifdef DEBUG_LEGION
        assert(sharded_view != NULL);
        assert(exchange != NULL);
        assert(record_valid);
#endif
        // All the users just need to do their registration
        RegionNode *node = runtime->forest->get_node(requirement.region);
        UpdateAnalysis *analysis = new UpdateAnalysis(runtime, this, 
                                      0/*index*/, version_info,
                                      requirement, node, mapped_instances,
                                      mapped_views,trace_info,init_precondition,
                                      termination_event, true/*track effects*/,
                                      false/*check initialized*/, record_valid,
                                      false/*skip output*/);
        analysis->add_reference();
        // Note that this call will clean up the analysis allocation
        effects_done = 
          runtime->forest->physical_perform_registration(analysis, 
              mapped_instances, trace_info, map_applied_conditions);
        // We need to fill in the sharded view before we do the next
        // call in case there are output effects due to restriction
        exchange->complete_exchange(this, sharded_view, 
                                    mapped_instances, map_applied_conditions);
        // We need everyone to be done mapping before we can do the overwrite
        if (!map_applied_conditions.empty())
        {
          Runtime::phase_barrier_arrive(inline_barrier, 1/*count*/,       
              Runtime::merge_events(map_applied_conditions));
          // No longer need this since one shard will wait on all of them
          map_applied_conditions.clear();
        }
        else
          Runtime::phase_barrier_arrive(inline_barrier, 1/*count*/);
        if (is_owner_shard)
        {
          // Wait for all the other shards to be done mapping first
          inline_barrier.wait(); 
          effects_done = 
              runtime->forest->overwrite_sharded(this, 0/*index*/, requirement,
                  sharded_view, version_info, trace_info, init_precondition, 
                  map_applied_conditions, false/*restrict*/);
        }
        Runtime::advance_barrier(inline_barrier);
      }
      else
      {
        // Everyone pretends like they are readers and does their 
        // separate updates as though they were going to just read 
        const bool is_write = IS_WRITE(requirement);
        if (is_write)
          requirement.privilege = LEGION_READ_ONLY; // pretend read-only for now
        UpdateAnalysis *analysis = NULL; 
        const RtEvent registration_precondition = 
          runtime->forest->physical_perform_updates(requirement, version_info,
              this, 0/*index*/, init_precondition, termination_event, 
              mapped_instances, trace_info, map_applied_conditions, analysis,
#ifdef DEBUG_LEGION
              get_logging_name(), unique_op_id,
#endif
              // No need to track effects since we know it can't be 
              // restricted in a control replicated context
              // Can't track initialized here because it might not be
              // correct with our altered privileges
              false/*track effects*/, record_valid/*record valid*/,
              false/*check initialized*/,
              // We can skip output for the same reason we don't 
              // need to track any effects
              true/*defer copies*/, true/*skip output*/); 
        // If we're a write, then switch back privileges
        if (is_write)
        {
          // In the read-write case we need to make sure everyone is done
          // performing their updates before anyone does a registration
          if (registration_precondition.exists())
            map_applied_conditions.insert(registration_precondition);
          if (!map_applied_conditions.empty())
          {
            Runtime::phase_barrier_arrive(inline_barrier, 1/*count*/,       
                Runtime::merge_events(map_applied_conditions));
            // Don't need these anymore since we're going to wait for them
            map_applied_conditions.clear();
          }
          else
            Runtime::phase_barrier_arrive(inline_barrier, 1/*count*/);
          // Set the privilege back to read-write
          requirement.privilege = LEGION_READ_WRITE;
          // Reset the usage of the analysis too
          analysis->usage = RegionUsage(requirement);
          // Wait for everyone to finish their updates
          inline_barrier.wait();
          // Advance the barrier to the next generation
          Runtime::advance_barrier(inline_barrier);
        }
        else
        {
          // In the read-only case we just need to wait for our registration
          // to be done before we can proceed
          if (registration_precondition.exists() && 
              !registration_precondition.has_triggered())
            registration_precondition.wait();
        }
        // Then do the registration, no need to track output effects since we
        // know that this instance can't be restricted in a control 
        // replicated context
        runtime->forest->physical_perform_registration(analysis, 
            mapped_instances, trace_info, map_applied_conditions);
        // If we have a write then we make a sharded view and 
        // then shard 0 will do the overwrite
        if (is_write)
        {
#ifdef DEBUG_LEGION
          assert(sharded_view != NULL);
          assert(exchange != NULL);
#endif
          // We need to fill in the sharded view before we do the next
          // call in case there are output effects due to restriction
          // Note this has to be done across all the shards in case 
          // the restricted copies go remote
          exchange->complete_exchange(this, sharded_view, 
                                      mapped_instances, map_applied_conditions);
          // We need everyone to be done mapping before we can do the overwrite
          if (!map_applied_conditions.empty())
          {
            Runtime::phase_barrier_arrive(inline_barrier, 1/*count*/,       
                Runtime::merge_events(map_applied_conditions));
            // No longer need this since one shard will wait on all of them
            map_applied_conditions.clear();
          }
          else
            Runtime::phase_barrier_arrive(inline_barrier, 1/*count*/);
          if (is_owner_shard)
          {
            // Wait for all the other shards to be done mapping first
            inline_barrier.wait();
            // Now we can do the replacement
            effects_done = 
              runtime->forest->overwrite_sharded(this, 0/*index*/, requirement,
                sharded_view, version_info, trace_info, init_precondition, 
                map_applied_conditions, false/*restrict*/);
          }
          Runtime::advance_barrier(inline_barrier);
        }
      }
#ifdef DEBUG_LEGION
      if (!IS_NO_ACCESS(requirement) && !requirement.privilege_fields.empty())
      {
        assert(!mapped_instances.empty());
        dump_physical_state(&requirement, 0);
      } 
#endif
      // Update our physical instance with the newly mapped instances
      // Have to do this before triggering the mapped event
      if (effects_done.exists())
      {
        region.impl->reset_references(mapped_instances, termination_event,
          Runtime::merge_events(&trace_info, init_precondition, effects_done));
      }
      else
        region.impl->reset_references(mapped_instances, termination_event,
                                      init_precondition);
      ApEvent map_complete_event = ApEvent::NO_AP_EVENT;
      if (mapped_instances.size() > 1)
      {
        std::set<ApEvent> mapped_events;
        for (unsigned idx = 0; idx < mapped_instances.size(); idx++)
          mapped_events.insert(mapped_instances[idx].get_ready_event());
        map_complete_event = Runtime::merge_events(&trace_info, mapped_events);
      }
      else if (!mapped_instances.empty())
        map_complete_event = mapped_instances[0].get_ready_event();
      if (runtime->legion_spy_enabled)
      {
        runtime->forest->log_mapping_decision(unique_op_id, parent_ctx,
                                              0/*idx*/, requirement,
                                              mapped_instances);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, map_complete_event,
                                        termination_event);
#endif
      }
      // See if we have any reservations to take as part of this map
      if (!atomic_locks.empty() || !arrive_barriers.empty())
      {
        if (!effects_done.exists())
          effects_done = 
            Runtime::merge_events(&trace_info, effects_done, termination_event);
        else
          effects_done = termination_event;
        // They've already been sorted in order 
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          map_complete_event = 
                Runtime::acquire_ap_reservation(it->first, it->second,
                                                map_complete_event);
          // We can also issue the release condition on our termination
          Runtime::release_reservation(it->first, effects_done);
        }
        for (std::vector<PhaseBarrier>::iterator it = 
              arrive_barriers.begin(); it != arrive_barriers.end(); it++)
        {
          if (runtime->legion_spy_enabled)
            LegionSpy::log_phase_barrier_arrival(unique_op_id, 
                                                 it->phase_barrier);
          Runtime::phase_barrier_arrive(it->phase_barrier, 1/*count*/,
                                        effects_done);    
        }
      }
      // Remove profiling our guard and trigger the profiling event if necessary
      int diff = -1; // need this dumbness for PGI
      if ((__sync_add_and_fetch(&outstanding_profiling_requests, diff) == 0) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
      // Now we can trigger the mapping event and indicate
      // to all our mapping dependences that we are mapped.
      RtEvent mapping_applied;
      if (!map_applied_conditions.empty())
        mapping_applied = Runtime::merge_events(map_applied_conditions);
      if (!acquired_instances.empty())
        mapping_applied = release_nonempty_acquired_instances(mapping_applied, 
                                                          acquired_instances);
      complete_mapping(complete_inline_mapping(mapping_applied));
      if (!map_complete_event.has_triggered())
      {
        // Issue a deferred trigger on our completion event
        // and mark that we are no longer responsible for 
        // triggering our completion event
        request_early_complete(map_complete_event);
        DeferredExecuteArgs deferred_execute_args(this);
        runtime->issue_runtime_meta_task(deferred_execute_args,
                                         LG_THROUGHPUT_DEFERRED_PRIORITY,
                                   Runtime::protect_event(map_complete_event));
      }
      else
        deferred_execute();
    }

    //--------------------------------------------------------------------------
    void ReplMapOp::activate(void)
    //--------------------------------------------------------------------------
    {
      MapOp::activate();
      exchange = NULL;
      view_did_broadcast = NULL;
      sharded_view = NULL;
    }

    //--------------------------------------------------------------------------
    void ReplMapOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_map_op();
      if (exchange != NULL)
        delete exchange;
      if (view_did_broadcast != NULL)
        delete view_did_broadcast;
      runtime->free_repl_map_op(this);
    }

    //--------------------------------------------------------------------------
    RtEvent ReplMapOp::complete_inline_mapping(RtEvent mapping_applied)
    //--------------------------------------------------------------------------
    {
      Runtime::phase_barrier_arrive(inline_barrier, 1/*count*/,mapping_applied);
      return inline_barrier;
    }

    /////////////////////////////////////////////////////////////
    // Repl Attach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplAttachOp::ReplAttachOp(Runtime *rt)
      : AttachOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplAttachOp::ReplAttachOp(const ReplAttachOp &rhs)
      : AttachOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplAttachOp::~ReplAttachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplAttachOp& ReplAttachOp::operator=(const ReplAttachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplAttachOp::initialize_replication(ReplicateContext *ctx,
                                              RtBarrier &resource_bar,
                                              ApBarrier &broadcast_bar,
                                              ApBarrier &reduce_bar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource_bar.exists());
      assert(exchange == NULL);
      assert(did_broadcast == NULL);
      assert(sharded_view == NULL);
#endif
      resource_barrier = resource_bar;
      ctx->advance_replicate_barrier(resource_bar, ctx->total_shards);
      broadcast_barrier = broadcast_bar;
      ctx->advance_replicate_barrier(broadcast_bar, 1/*arrivals*/);
      // No matter what we're going to need a view broadcast either to make
      // an instance which everyone has the name of or a sharded view
      did_broadcast = 
          new ValueBroadcast<DistributedID>(ctx, 0/*owner*/, COLLECTIVE_LOC_77);
      if ((resource == LEGION_EXTERNAL_INSTANCE) || local_files)
      {
        // In this case we need a second generation of the resource_bar
        ctx->advance_replicate_barrier(resource_bar, ctx->total_shards);
        exchange = new ShardedMappingExchange(COLLECTIVE_LOC_78, ctx,
                           ctx->owner_shard->shard_id, false/*perform checks*/);
        
        // if we're shard 0 then get the distributed id and send it out
        if (ctx->owner_shard->shard_id == 0)
        {
          DistributedID view_did = runtime->get_available_distributed_id();
          // make it and register it with the runtime
          sharded_view = new ShardedView(runtime->forest,
            view_did, runtime->address_space, true/*register now*/);
          // then broadcast the result out so the other nodes can grab it
          did_broadcast->broadcast(sharded_view->did);
        }
      }
      else
      {
        reduce_barrier = reduce_bar;
        ctx->advance_replicate_barrier(reduce_bar, ctx->total_shards);
      }
    }

    //--------------------------------------------------------------------------
    void ReplAttachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_attach_op();
      resource_barrier = RtBarrier::NO_RT_BARRIER;
      repl_mapping_applied = RtUserEvent::NO_RT_USER_EVENT;
      exchange = NULL;
      did_broadcast = NULL;
      sharded_view = NULL;
      all_mapped_event = RtEvent::NO_RT_EVENT;
      exchange_complete = false;
    }

    //--------------------------------------------------------------------------
    void ReplAttachOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_attach_op();
      if (exchange != NULL)
        delete exchange;
      if (did_broadcast != NULL)
        delete did_broadcast;
      runtime->free_repl_attach_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplAttachOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;  
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      const bool owner_shard = (repl_ctx->owner_shard->shard_id == 0);
      if (!owner_shard)
      {
        if ((resource == LEGION_EXTERNAL_INSTANCE) || local_files)
        {
          // Get the distributed ID for the sharded view and request it
          const DistributedID sharded_did = did_broadcast->get_value();
          RtEvent ready;
          sharded_view = static_cast<ShardedView*>(
              runtime->find_or_request_logical_view(sharded_did, ready));
          if (ready.exists())
            preconditions.insert(ready);
        }
      }
      else // Only need the version info on the owner node
        runtime->forest->perform_versioning_analysis(this, 0/*idx*/,
                                                     requirement,
                                                     version_info,
                                                     preconditions);
      if (!preconditions.empty())
        enqueue_ready_operation(Runtime::merge_events(preconditions));
      else
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void ReplAttachOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      const bool is_owner_shard = (repl_ctx->owner_shard->shard_id == 0);
      if ((resource == LEGION_EXTERNAL_INSTANCE) || local_files)
      {
#ifdef DEBUG_LEGION
        assert(!restricted);
        assert(exchange != NULL);
        assert(sharded_view != NULL);
#endif
        switch (resource)
        {
          case LEGION_EXTERNAL_POSIX_FILE:
          case LEGION_EXTERNAL_HDF5_FILE:
            {
              external_instance = 
                runtime->forest->create_external_instance(this, requirement, 
                                                requirement.instance_fields);
              break;
            }
          case LEGION_EXTERNAL_INSTANCE:
            {
              external_instance = 
                runtime->forest->create_external_instance(this, requirement,
                          layout_constraint_set.field_constraint.field_set);
              break;
            }
          default:
            assert(false);
        }
        InstanceSet attach_instances(1);
        attach_instances[0] = external_instance;
        InnerContext *context = find_physical_context(0/*index*/, requirement);
        std::vector<InstanceView*> attach_views;
        context->convert_target_views(attach_instances, attach_views);
        exchange->initiate_exchange(attach_instances, attach_views);
        // Once we're ready to map we can tell the memory manager that
        // this instance can be safely acquired for use
        IndividualManager *external_manager = 
          external_instance.get_instance_manager()->as_individual_manager();
        MemoryManager *memory_manager = external_manager->memory_manager;
        memory_manager->attach_external_instance(external_manager);
        RegionNode *node = runtime->forest->get_node(requirement.region);
        ApUserEvent termination_event;
        if (mapping)
          termination_event = Runtime::create_ap_user_event(NULL);
        const PhysicalTraceInfo trace_info(this, 0/*idx*/, true/*init*/);
        UpdateAnalysis *analysis = new UpdateAnalysis(runtime, this, 0/*index*/,
          version_info, requirement, node, attach_instances, attach_views,
          trace_info, ApEvent::NO_AP_EVENT, mapping ? termination_event : 
            completion_event, false/*track effects*/, 
          false/*check initialized*/, true/*record valid*/,true/*skip output*/);
        analysis->add_reference();
        // Have each operation do its own registration
        // Note this will clean up the analysis allocation above
        runtime->forest->physical_perform_registration(analysis, 
            attach_instances, trace_info, map_applied_conditions);
        exchange->complete_exchange(this, sharded_view, 
                                    attach_instances, map_applied_conditions);
        // Make sure all these are done before we do the overwrite
        if (!map_applied_conditions.empty())
        {
          Runtime::phase_barrier_arrive(resource_barrier, 1/*count*/,       
              Runtime::merge_events(map_applied_conditions));
          // No longer need this since one shard will wait on all of them
          map_applied_conditions.clear();
        }
        else
          Runtime::phase_barrier_arrive(resource_barrier, 1/*count*/);
        if (is_owner_shard)
        {
          // Wait for all the other shards to be done mapping first
          resource_barrier.wait();
          // Now we can do the replacement
          const ApEvent attach_event = 
            runtime->forest->overwrite_sharded(this, 0/*index*/, requirement,
                    sharded_view, version_info, trace_info,
                    ApEvent::NO_AP_EVENT, map_applied_conditions, restricted);
          Runtime::phase_barrier_arrive(broadcast_barrier, 1/*count*/, 
                                        attach_event);
        }
        Runtime::advance_barrier(resource_barrier);
#ifdef DEBUG_LEGION
        assert(external_instance.has_ref());
#endif
        // This operation is ready once the file is attached
        if (mapping)
        {
          attach_instances[0].set_ready_event(broadcast_barrier);
          region.impl->reset_references(attach_instances, termination_event,
                                        broadcast_barrier);
        }
        else
          region.impl->set_reference(external_instance);
        // Also set the sharded view in this case
        region.impl->set_sharded_view(sharded_view);
        // Make sure that all the attach operations are done mapping
        // before we consider this attach operation done
        if (!map_applied_conditions.empty())
          Runtime::phase_barrier_arrive(resource_barrier, 1/*count*/,
                        Runtime::merge_events(map_applied_conditions));
        else
          Runtime::phase_barrier_arrive(resource_barrier, 1/*count*/);
        complete_mapping(resource_barrier);
        request_early_complete(broadcast_barrier);
        complete_execution(Runtime::protect_event(broadcast_barrier));
      }
      else
      {
        ApUserEvent termination_event;
        if (mapping)
        {
          termination_event = Runtime::create_ap_user_event(NULL);
          Runtime::phase_barrier_arrive(reduce_barrier, 1/*count*/,
                                        termination_event);
        }
        if (is_owner_shard)
        {
          // Make our instance now and send out the DID
          switch (resource)
          {
            case LEGION_EXTERNAL_POSIX_FILE:
            case LEGION_EXTERNAL_HDF5_FILE:
              {
                external_instance = 
                  runtime->forest->create_external_instance(this, requirement, 
                                                  requirement.instance_fields);
                break;
              }
              // No external instances here by definition
            default:
              assert(false);
          }
          
          InstanceSet attach_instances(1);
          attach_instances[0] = external_instance;
          // Once we're ready to map we can tell the memory manager that
          // this instance can be safely acquired for use
          IndividualManager *external_manager = 
            external_instance.get_instance_manager()->as_individual_manager();
          MemoryManager *memory_manager = external_manager->memory_manager;
          memory_manager->attach_external_instance(external_manager);
          // We can't broadcast the DID until after doing the attach
          // to the memory in case we update the reference state
          did_broadcast->broadcast(external_instance.get_manager()->did);
          const PhysicalTraceInfo trace_info(this, 0/*idx*/, true/*init*/);
          InnerContext *context = find_physical_context(0/*index*/,requirement);
          std::vector<InstanceView*> attach_views;
          context->convert_target_views(attach_instances, attach_views);
#ifdef DEBUG_LEGION
          assert(attach_views.size() == 1);
#endif
          ApEvent attach_event = runtime->forest->attach_external(this,0/*idx*/,
                                                        requirement,
                                                        attach_views[0],
                                                        attach_views[0],
                                                        mapping ?
                                                         (ApEvent)reduce_barrier
                                                         : completion_event,
                                                        version_info,
                                                        trace_info,
                                                        map_applied_conditions,
                                                        restricted);
#ifdef DEBUG_LEGION
          assert(external_instance.has_ref());
#endif
          Runtime::phase_barrier_arrive(broadcast_barrier, 1/*count*/,
                                        attach_event);
          // Save the instance information out to region
          if (mapping)
          {
            attach_instances[0].set_ready_event(broadcast_barrier);
            region.impl->reset_references(attach_instances, termination_event,
                                          broadcast_barrier);
          }
          else
            region.impl->set_reference(external_instance);
          // This operation is ready once the file is attached
          // Make sure that all the attach operations are done mapping
          // before we consider this attach operation done
          if (!map_applied_conditions.empty())
            Runtime::phase_barrier_arrive(resource_barrier, 1/*count*/,
                          Runtime::merge_events(map_applied_conditions));
          else
            Runtime::phase_barrier_arrive(resource_barrier, 1/*count*/);
          complete_mapping(resource_barrier);
          request_early_complete(broadcast_barrier);
          complete_execution(Runtime::protect_event(broadcast_barrier));
        }
        else
        {
          FieldSpaceNode *node = 
            runtime->forest->get_node(requirement.region.get_field_space());
          FieldMask instance_fields = 
            node->get_field_mask(requirement.privilege_fields);
          // Get the DID for the common manager and request it
          DistributedID manager_did = did_broadcast->get_value();
          RtEvent ready;
          PhysicalManager *manager = 
              runtime->find_or_request_instance_manager(manager_did, ready);
          // Wait for the manager to be ready 
          if (ready.exists())
            ready.wait();
          external_instance = InstanceRef(manager, instance_fields);
          // Save the instance information out to region
          if (mapping)
          {
            InstanceSet attach_instances(1);
            attach_instances[0] = external_instance;
            attach_instances[0].set_ready_event(broadcast_barrier);
            region.impl->reset_references(attach_instances, termination_event,
                                          broadcast_barrier);
          }
          else
            region.impl->set_reference(external_instance);
          // Record that we're mapped once everyone else does
          Runtime::phase_barrier_arrive(resource_barrier, 1/*count*/);
          complete_mapping(resource_barrier);
          complete_execution(Runtime::protect_event(broadcast_barrier));
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Repl Detach Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplDetachOp::ReplDetachOp(Runtime *rt)
      : DetachOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplDetachOp::ReplDetachOp(const ReplDetachOp &rhs)
      : DetachOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplDetachOp::~ReplDetachOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplDetachOp& ReplDetachOp::operator=(const ReplDetachOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::initialize_replication(ReplicateContext *ctx,
                                              RtBarrier &resource_bar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(resource_bar.exists());
#endif
      resource_barrier = resource_bar;
      ctx->advance_replicate_barrier(resource_bar, ctx->total_shards);
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_detach_op();
      resource_barrier = RtBarrier::NO_RT_BARRIER;
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_detach_op();
      runtime->free_repl_detach_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      const bool is_owner_shard = (repl_ctx->owner_shard->shard_id == 0);
      const PhysicalTraceInfo trace_info(this, 0/*index*/, true/*init*/);
      // Actual unmap of an inline mapped region was deferred to here
      if (region.impl->is_mapped())
        region.impl->unmap_region();
      // Now we can get the reference we need for the detach operation
      InstanceSet references;
      region.impl->get_references(references);
#ifdef DEBUG_LEGION
      assert(references.size() == 1);
#endif
      InstanceRef reference = references[0];
      // Check that this is actually a file
      PhysicalManager *manager = reference.get_instance_manager();
#ifdef DEBUG_LEGION
      assert(!manager->is_reduction_manager()); 
#endif
      ShardedView *sharded_view = region.impl->get_sharded_view();
      ApEvent detach_event;
      if ((sharded_view != NULL) || (is_owner_shard))
      {
        // Everybody does registration and filtering in the case
        // where there is a sharded view because there are different 
        // instances for each shard
        // Only the owner does it in the case where there isn't a
        // sharded view because there is only one instance for all shards
        InnerContext *context = find_physical_context(0/*index*/, requirement);
        std::vector<InstanceView*> inst_views;
        context->convert_target_views(references, inst_views);
        detach_event = runtime->forest->detach_external(requirement,
            this, 0/*index*/, version_info, inst_views[0],
            trace_info, map_applied_conditions, sharded_view);
        // Also tell the runtime to detach the external instance from memory
        // This has to be done before we can consider this mapped
        RtEvent detached_event = manager->detach_external_instance();
        if (detached_event.exists())
          map_applied_conditions.insert(detached_event);
        if (runtime->legion_spy_enabled)
        {
          runtime->forest->log_mapping_decision(unique_op_id, parent_ctx,
                                      0/*idx*/, requirement, references);
#ifdef LEGION_SPY
          LegionSpy::log_operation_events(unique_op_id, detach_event,
                                          completion_event);
#endif
        }
      }
      // Make sure that all the detach operations are done before 
      // we count any of them as being mapped
      if (!map_applied_conditions.empty())
        Runtime::phase_barrier_arrive(resource_barrier, 1/*count*/,
                      Runtime::merge_events(map_applied_conditions));
      else
        Runtime::phase_barrier_arrive(resource_barrier, 1/*count*/);
      complete_mapping(resource_barrier);

      request_early_complete(detach_event);
      complete_execution(Runtime::protect_event(detach_event));
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::select_sources(const unsigned index,
                                      const InstanceRef &target,
                                      const InstanceSet &sources,
                                      std::vector<unsigned> &ranking)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index == 0);
#endif
      // Pick any instances other than external ones
      std::vector<unsigned> remote_ranking;
      for (unsigned idx = 0; idx < sources.size(); idx++)
      {
        const InstanceRef &ref = sources[idx];
        PhysicalManager *manager = ref.get_instance_manager();
        if (manager->is_external_instance())
          continue;
        if (manager->owner_space == runtime->address_space)
          ranking.push_back(idx);
        else
          remote_ranking.push_back(idx);
      }
      if (!remote_ranking.empty())
        ranking.insert(ranking.end(), 
                       remote_ranking.begin(), remote_ranking.end());
    }

    //--------------------------------------------------------------------------
    void ReplDetachOp::record_unordered_kind(
          std::map<std::pair<LogicalRegion,FieldID>,ReplDetachOp*> &detachments)
    //--------------------------------------------------------------------------
    {
      const RegionRequirement &req = region.impl->get_requirement();
#ifdef DEBUG_LEGION
      assert(!req.privilege_fields.empty());
#endif
      const std::pair<LogicalRegion,FieldID> key(req.region,
          *(req.privilege_fields.begin()));
#ifdef DEBUG_LEGION
      assert(detachments.find(key) == detachments.end());
#endif
      detachments[key] = this; 
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceOp::ReplTraceOp(Runtime *rt)
      : ReplFenceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceOp::ReplTraceOp(const ReplTraceOp &rhs)
      : ReplFenceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplTraceOp::~ReplTraceOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceOp& ReplTraceOp::operator=(const ReplTraceOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplTraceOp::execute_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping_tracker == NULL);
#endif
      // Make a dependence tracker
      mapping_tracker = new MappingDependenceTracker();
      // See if we have any fence dependences
      execution_fence_event = parent_ctx->register_implicit_dependences(this);
      parent_ctx->invalidate_trace_cache(local_trace, this);

      trigger_dependence_analysis();
      end_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplTraceOp::sync_for_replayable_check(void)
    //--------------------------------------------------------------------------
    {
      // Should only be called by derived classes
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool ReplTraceOp::exchange_replayable(ReplicateContext *ctx,bool replayable)
    //--------------------------------------------------------------------------
    {
      // Should only be called by derived classes
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void ReplTraceOp::elide_fences_pre_sync(void)
    //--------------------------------------------------------------------------
    {
      // Should only be called by derived classes
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ReplTraceOp::elide_fences_post_sync(void)
    //--------------------------------------------------------------------------
    {
      // Should only be called by derived classes
      assert(false);
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceCaptureOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceCaptureOp::ReplTraceCaptureOp(Runtime *rt)
      : ReplTraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceCaptureOp::ReplTraceCaptureOp(const ReplTraceCaptureOp &rhs)
      : ReplTraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplTraceCaptureOp::~ReplTraceCaptureOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceCaptureOp& ReplTraceCaptureOp::operator=(
                                                  const ReplTraceCaptureOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplTraceCaptureOp::initialize_capture(ReplicateContext *ctx, 
                                          bool has_block, bool remove_trace_ref)
    //--------------------------------------------------------------------------
    {
      initialize_repl_fence(ctx, EXECUTION_FENCE, false/*need future*/);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      tracing = false;
      current_template = NULL;
      has_blocking_call = has_block;
      remove_trace_reference = remove_trace_ref;
      // Get a collective ID to use for check all replayable
      replayable_collective_id = 
        ctx->get_next_collective_index(COLLECTIVE_LOC_85); 
      replay_sync_collective_id =
        ctx->get_next_collective_index(COLLECTIVE_LOC_91);
      pre_elide_fences_collective_id =
        ctx->get_next_collective_index(COLLECTIVE_LOC_92);
      post_elide_fences_collective_id =
        ctx->get_next_collective_index(COLLECTIVE_LOC_93);
    }

    //--------------------------------------------------------------------------
    void ReplTraceCaptureOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      current_template = NULL;
      replayable_collective_id = 0;
      has_blocking_call = false;
      remove_trace_reference = false;
    }

    //--------------------------------------------------------------------------
    void ReplTraceCaptureOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_repl_capture_op(this);
    }

    //--------------------------------------------------------------------------
    const char* ReplTraceCaptureOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_CAPTURE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind ReplTraceCaptureOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_CAPTURE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void ReplTraceCaptureOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(local_trace != NULL);
#endif
      // Indicate that we are done capturing this trace
      local_trace->end_trace_capture();
      // Register this fence with all previous users in the parent's context
      ReplFenceOp::trigger_dependence_analysis();
      parent_ctx->record_previous_trace(local_trace);
      if (local_trace->is_recording())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif
        current_template = physical_trace->get_current_template();
        physical_trace->record_previous_template_completion(
            get_completion_event());
        physical_trace->clear_cached_template();
      }
    }

    //--------------------------------------------------------------------------
    void ReplTraceCaptureOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Now finish capturing the physical trace
      if (local_trace->is_recording())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
        assert(current_template != NULL);
        assert(local_trace->get_physical_trace() != NULL);
        assert(current_template->is_recording());
#endif
        current_template->finalize(has_blocking_call, this);
        if (!current_template->is_replayable())
        {
          physical_trace->record_failed_capture(current_template);
          ApEvent pending_deletion;
          if (!current_template->defer_template_deletion(pending_deletion,
                                                  map_applied_conditions))
            delete current_template;
          if (pending_deletion.exists())
            execution_preconditions.insert(pending_deletion);
        }
        else
          physical_trace->record_replayable_capture(current_template);
        // Reset the local trace
        local_trace->initialize_tracing_state();
      }
      if (remove_trace_reference && local_trace->remove_reference())
        delete local_trace;
      ReplFenceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    void ReplTraceCaptureOp::sync_for_replayable_check(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      SlowBarrier replay_sync_barrier(repl_ctx, replay_sync_collective_id);
      replay_sync_barrier.perform_collective_sync();
    }

    //--------------------------------------------------------------------------
    bool ReplTraceCaptureOp::exchange_replayable(ReplicateContext *repl_ctx,
                                                 bool shard_replayable)
    //--------------------------------------------------------------------------
    {
      // Check to see if this template is replayable across all the shards
      AllReduceCollective<ProdReduction<bool> > 
        all_replayable_collective(repl_ctx, replayable_collective_id);
      return all_replayable_collective.sync_all_reduce(shard_replayable);
    }

    //--------------------------------------------------------------------------
    void ReplTraceCaptureOp::elide_fences_pre_sync(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      SlowBarrier pre_sync_barrier(repl_ctx, pre_elide_fences_collective_id);
      pre_sync_barrier.perform_collective_sync();
    }

    //--------------------------------------------------------------------------
    void ReplTraceCaptureOp::elide_fences_post_sync(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      SlowBarrier post_sync_barrier(repl_ctx, post_elide_fences_collective_id);
      post_sync_barrier.perform_collective_sync();
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceCompleteOp 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceCompleteOp::ReplTraceCompleteOp(Runtime *rt)
      : ReplTraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceCompleteOp::ReplTraceCompleteOp(const ReplTraceCompleteOp &rhs)
      : ReplTraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplTraceCompleteOp::~ReplTraceCompleteOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceCompleteOp& ReplTraceCompleteOp::operator=(
                                                 const ReplTraceCompleteOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::initialize_complete(ReplicateContext *ctx, 
                                                  bool has_block)
    //--------------------------------------------------------------------------
    {
      initialize_repl_fence(ctx, EXECUTION_FENCE, false/*need future*/);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      // Now mark our trace as NULL to avoid registering this operation
      trace = NULL;
      tracing = false;
      current_template = NULL;
      template_completion = ApEvent::NO_AP_EVENT;
      replayed = false;
      has_blocking_call = has_block;
      // Get a collective ID to use for check all replayable
      replayable_collective_id = 
        ctx->get_next_collective_index(COLLECTIVE_LOC_86);
      replay_sync_collective_id =
        ctx->get_next_collective_index(COLLECTIVE_LOC_91);
      pre_elide_fences_collective_id =
        ctx->get_next_collective_index(COLLECTIVE_LOC_92);
      post_elide_fences_collective_id =
        ctx->get_next_collective_index(COLLECTIVE_LOC_93);
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      current_template = NULL;
      template_completion = ApEvent::NO_AP_EVENT;
      replayable_collective_id = 0;
      replayed = false;
      has_blocking_call = false;
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_repl_trace_op(this);
    }

    //--------------------------------------------------------------------------
    const char* ReplTraceCompleteOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_COMPLETE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind ReplTraceCompleteOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_COMPLETE_OP_KIND; 
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(local_trace != NULL);
#endif
#ifdef LEGION_SPY
      if (local_trace->is_replaying())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif
        local_trace->perform_logging(
         physical_trace->get_current_template()->get_fence_uid(), unique_op_id);
      }
#endif
      local_trace->end_trace_execution(this);
      parent_ctx->record_previous_trace(local_trace);

      if (local_trace->is_replaying())
      {
        if (has_blocking_call)
          REPORT_LEGION_ERROR(ERROR_INVALID_PHYSICAL_TRACING,
            "Physical tracing violation! Trace %d in task %s (UID %lld) "
            "encountered a blocking API call that was unseen when it was "
            "recorded. It is required that traces do not change their "
            "behavior.", local_trace->get_trace_id(),
            parent_ctx->get_task_name(), parent_ctx->get_unique_id())
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif
        current_template = physical_trace->get_current_template();
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
#endif
        parent_ctx->update_current_fence(this, true, true);
        physical_trace->record_previous_template_completion(completion_event);
        local_trace->initialize_tracing_state();
        replayed = true;
        return;
      }
      else if (local_trace->is_recording())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
#endif
        current_template = physical_trace->get_current_template();
        physical_trace->record_previous_template_completion(completion_event);
        physical_trace->clear_cached_template();
      }

      // If this is a static trace, then we remove our reference when we're done
      if (local_trace->is_static_trace())
      {
        StaticTrace *static_trace = static_cast<StaticTrace*>(local_trace);
        if (static_trace->remove_reference())
          delete static_trace;
      }
      ReplFenceOp::trigger_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (replayed)
      {
        // Having all our mapping dependences satisfied means that the previous 
        // replay of this template is done so we can start ours now
        std::set<RtEvent> replayed_events;
        current_template->perform_replay(runtime, replayed_events);
        if (!replayed_events.empty())
        {
          enqueue_ready_operation(Runtime::merge_events(replayed_events));
          return;
        }
      }
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Now finish capturing the physical trace
      if (local_trace->is_recording())
      {
        PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
        assert(physical_trace != NULL);
        assert(current_template != NULL);
        assert(local_trace->get_physical_trace() != NULL);
        assert(current_template->is_recording());
#endif
        current_template->finalize(has_blocking_call, this);
        if (!current_template->is_replayable())
        {
          physical_trace->record_failed_capture(current_template);
          ApEvent pending_deletion;
          if (!current_template->defer_template_deletion(pending_deletion,
                                                  map_applied_conditions))
            delete current_template;
          if (pending_deletion.exists())
            execution_preconditions.insert(pending_deletion);
        }
        else
          physical_trace->record_replayable_capture(current_template);
        local_trace->initialize_tracing_state();
      }
      else if (replayed)
      { 
#ifdef DEBUG_LEGION
        assert(current_template != NULL);
#endif
        std::set<ApEvent> template_postconditions;
        current_template->finish_replay(template_postconditions);
        // Do our arrival on the mapping fence
        Runtime::phase_barrier_arrive(mapping_fence_barrier, 1/*count*/);
        complete_mapping(mapping_fence_barrier);
        if (!template_postconditions.empty())
          Runtime::phase_barrier_arrive(execution_fence_barrier, 1/*count*/,
              Runtime::merge_events(NULL, template_postconditions));
        else
          Runtime::phase_barrier_arrive(execution_fence_barrier, 1/*count*/);
        Runtime::trigger_event(NULL, completion_event, execution_fence_barrier);
        need_completion_trigger = false;
        complete_execution();
        return;
      }
      ReplFenceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::sync_for_replayable_check(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      SlowBarrier replay_sync_barrier(repl_ctx, replay_sync_collective_id);
      replay_sync_barrier.perform_collective_sync();
    }

    //--------------------------------------------------------------------------
    bool ReplTraceCompleteOp::exchange_replayable(ReplicateContext *repl_ctx,
                                                  bool shard_replayable)
    //--------------------------------------------------------------------------
    {
      // Check to see if this template is replayable across all the shards
      AllReduceCollective<ProdReduction<bool> > 
        all_replayable_collective(repl_ctx, replayable_collective_id);
      return all_replayable_collective.sync_all_reduce(shard_replayable);
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::elide_fences_pre_sync(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      SlowBarrier pre_sync_barrier(repl_ctx, pre_elide_fences_collective_id);
      pre_sync_barrier.perform_collective_sync();
    }

    //--------------------------------------------------------------------------
    void ReplTraceCompleteOp::elide_fences_post_sync(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      SlowBarrier post_sync_barrier(repl_ctx, post_elide_fences_collective_id);
      post_sync_barrier.perform_collective_sync();
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceReplayOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceReplayOp::ReplTraceReplayOp(Runtime *rt)
      : ReplTraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceReplayOp::ReplTraceReplayOp(const ReplTraceReplayOp &rhs)
      : ReplTraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplTraceReplayOp::~ReplTraceReplayOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceReplayOp& ReplTraceReplayOp::operator=(
                                                   const ReplTraceReplayOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplTraceReplayOp::initialize_replay(ReplicateContext *ctx, 
                                              LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      initialize_repl_fence(ctx, EXECUTION_FENCE, false/*need future*/);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      for (int idx = 0; idx < TRACE_SELECTION_ROUNDS; idx++)
        trace_selection_collective_ids[idx] = 
          ctx->get_next_collective_index(COLLECTIVE_LOC_87);
    }

    //--------------------------------------------------------------------------
    void ReplTraceReplayOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void ReplTraceReplayOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_repl_replay_op(this);
    }

    //--------------------------------------------------------------------------
    const char* ReplTraceReplayOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_REPLAY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind ReplTraceReplayOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_REPLAY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void ReplTraceReplayOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(trace == NULL);
      assert(local_trace != NULL);
#endif
      PhysicalTrace *physical_trace = local_trace->get_physical_trace();
#ifdef DEBUG_LEGION
      assert(physical_trace != NULL);
#endif
      bool recurrent = true;
      bool fence_registered = false;
      bool is_recording = local_trace->is_recording();
      if ((physical_trace->get_current_template() == NULL) || is_recording)
      {
        recurrent = false;
        {
          // Wait for the previous recordings to be done before checking
          // template preconditions, otherwise no template would exist.
          RtEvent mapped_event = parent_ctx->get_current_mapping_fence_event();
          if (mapped_event.exists())
            mapped_event.wait();
        }
#ifdef DEBUG_LEGION
        assert(!(local_trace->is_recording() || local_trace->is_replaying()));
        ReplicateContext *repl_ctx =dynamic_cast<ReplicateContext*>(parent_ctx);
        assert(repl_ctx != NULL);
#else
        ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif

        if (physical_trace->get_current_template() == NULL)
        {
          int selected_template_index = -2;  
          std::vector<int> viable_templates;
          for (int round = 0; round < TRACE_SELECTION_ROUNDS; round++)
          {
            // Exponential back-off: the more rounds we go, the
            // more templates we try to find to build consensus
            const unsigned number_to_find = 1 << round;
            if ((viable_templates.empty() || (viable_templates.back() >= 0)) &&
                physical_trace->find_viable_templates(this, 
                                                      map_applied_conditions,
                                                      number_to_find, 
                                                      viable_templates))
            {
              // If we checked all the templates figure out what kind of 
              // guard to add:
              // Use -1 to indicate that we're done but have viable templates
              // Use -2 to indicate we have no viable templates
              if (!viable_templates.empty())
                viable_templates.push_back(-1);
              else
                viable_templates.push_back(-2);
            }
#ifdef DEBUG_LEGION
            assert(!viable_templates.empty());
#endif
            // Perform an exchange to see if we have consensus
            TemplateIndexExchange index_exchange(repl_ctx, 
                    trace_selection_collective_ids[round]);
            index_exchange.initiate_exchange(viable_templates);
            std::map<int/*index*/,unsigned/*count*/> result_templates;
            index_exchange.complete_exchange(result_templates);
            // First, if we have at least one shard that says that it
            // has no viable templates then we're done
            if (result_templates.find(-2) == result_templates.end())
            {
              // Otherwise go through in reverse order and look for one that
              // has consensus from all the shards
              const size_t total_shards = repl_ctx->shard_manager->total_shards;
              for (std::map<int,unsigned>::reverse_iterator rit = 
                    result_templates.rbegin(); rit != 
                    result_templates.rend(); rit++)
              {
#ifdef DEBUG_LEGION
                assert(rit->second <= total_shards);
#endif
                // If we have a template that is viable for all the shards
                // then we've succesffully identified a template to use
                if (rit->second == total_shards)
                {
                  // Note this could also be -1 in the case were all
                  // the shards have identified all their viable templates
                  selected_template_index = rit->first;
                  break;
                }
              }
            }
            else
              selected_template_index = -1;
            // If we picked an index then we're done
            if (selected_template_index != -2)
              break;
          }
          // If we successfully identified a template for all the shards
          // to use then we record that in the trace 
          if (selected_template_index >= 0)
            physical_trace->select_template(selected_template_index);
        }
#ifdef DEBUG_LEGION
        assert(physical_trace->get_current_template() == NULL ||
               !physical_trace->get_current_template()->is_recording());
#endif
        parent_ctx->perform_fence_analysis(this, execution_preconditions,
                                           true/*mapping*/, true/*execution*/);
        physical_trace->set_current_execution_fence_event(
            get_completion_event());
        fence_registered = true;
      }

      if (physical_trace->get_current_template() != NULL)
      {
        // If we're recurrent, then check to see if we had any intermeidate
        // ops for which we still need to perform the fence analysis
        if (recurrent && local_trace->has_intermediate_operations())
        {
          parent_ctx->perform_fence_analysis(this, execution_preconditions,
                                       true/*mapping*/, true/*execution*/);
          local_trace->reset_intermediate_operations();
        }
        if (!fence_registered)
          execution_preconditions.insert(
              parent_ctx->get_current_execution_fence_event());
        ApEvent fence_completion = recurrent ?
          physical_trace->get_previous_template_completion() : 
          get_completion_event();
        physical_trace->initialize_template(fence_completion, recurrent);
        local_trace->set_state_replay();
#ifdef LEGION_SPY
        physical_trace->get_current_template()->set_fence_uid(unique_op_id);
#endif
      }
      else if (!fence_registered)
      {
        parent_ctx->perform_fence_analysis(this, execution_preconditions,
                                           true/*mapping*/, true/*execution*/);
        physical_trace->set_current_execution_fence_event(
            get_completion_event());
      }

      // Now update the parent context with this fence before we can complete
      // the dependence analysis and possibly be deactivated
      parent_ctx->update_current_fence(this, true, true);
    }

    //--------------------------------------------------------------------------
    void ReplTraceReplayOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceBeginOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceBeginOp::ReplTraceBeginOp(Runtime *rt)
      : ReplTraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceBeginOp::ReplTraceBeginOp(const ReplTraceBeginOp &rhs)
      : ReplTraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplTraceBeginOp::~ReplTraceBeginOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceBeginOp& ReplTraceBeginOp::operator=(const ReplTraceBeginOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplTraceBeginOp::initialize_begin(ReplicateContext *ctx, 
                                            LegionTrace *trace)
    //--------------------------------------------------------------------------
    {
      initialize_repl_fence(ctx, MAPPING_FENCE, false/*need future*/);
#ifdef DEBUG_LEGION
      assert(trace != NULL);
#endif
      local_trace = trace;
      trace = NULL;
      tracing = false;
    }

    //--------------------------------------------------------------------------
    void ReplTraceBeginOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
    }

    //--------------------------------------------------------------------------
    void ReplTraceBeginOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_operation();
      runtime->free_repl_begin_op(this);
    }

    //--------------------------------------------------------------------------
    const char* ReplTraceBeginOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_BEGIN_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind ReplTraceBeginOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_BEGIN_OP_KIND;
    }

    /////////////////////////////////////////////////////////////
    // ReplTraceSummaryOp
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTraceSummaryOp::ReplTraceSummaryOp(Runtime *rt)
      : ReplTraceOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceSummaryOp::ReplTraceSummaryOp(const ReplTraceSummaryOp &rhs)
      : ReplTraceOp(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplTraceSummaryOp::~ReplTraceSummaryOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplTraceSummaryOp& ReplTraceSummaryOp::operator=(
                                                  const ReplTraceSummaryOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplTraceSummaryOp::initialize_summary(ReplicateContext *ctx,
                                                ShardedPhysicalTemplate *tpl,
                                                Operation *invalidator)
    //--------------------------------------------------------------------------
    {
      // Do NOT call initialize_repl_fence here, we're in the dependence
      // analysis stage of the pipeline and we need to get our mapping
      // fence from a different location to avoid racing with the application
      initialize(ctx, MAPPING_FENCE, false/*need future*/, false/*track*/);
      mapping_fence_barrier = ctx->get_next_summary_fence_barrier();
      context_index = invalidator->get_ctx_index();
      current_template = tpl;
      // The summary could have been marked as being traced,
      // so here we forcibly clear them out.
      trace = NULL;
      tracing = false;
    }

    //--------------------------------------------------------------------------
    void ReplTraceSummaryOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_operation();
      current_template = NULL;
    }

    //--------------------------------------------------------------------------
    void ReplTraceSummaryOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_fence();
      runtime->free_repl_summary_op(this);
    }

    //--------------------------------------------------------------------------
    const char* ReplTraceSummaryOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TRACE_SUMMARY_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind ReplTraceSummaryOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TRACE_SUMMARY_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void ReplTraceSummaryOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_fence_analysis(true/*register fence also*/);
    }

    //--------------------------------------------------------------------------
    void ReplTraceSummaryOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void ReplTraceSummaryOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      if (current_template->is_replayable())
        current_template->apply_postcondition(this, map_applied_conditions);
      ReplFenceOp::trigger_mapping();
    }

    //--------------------------------------------------------------------------
    void ReplTraceSummaryOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
    }

    /////////////////////////////////////////////////////////////
    // Shard Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardMapping::ShardMapping(void)
      : Collectable()
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShardMapping::ShardMapping(const ShardMapping &rhs)
      : Collectable()
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShardMapping::ShardMapping(const std::vector<AddressSpaceID> &spaces)
      : Collectable(), address_spaces(spaces)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShardMapping::~ShardMapping(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShardMapping& ShardMapping::operator=(const ShardMapping &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    AddressSpaceID ShardMapping::operator[](unsigned idx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < address_spaces.size());
#endif
      return address_spaces[idx];
    }

    //--------------------------------------------------------------------------
    AddressSpaceID& ShardMapping::operator[](unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < address_spaces.size());
#endif
      return address_spaces[idx];
    }

    //--------------------------------------------------------------------------
    void ShardMapping::pack_mapping(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(address_spaces.size());
      for (std::vector<AddressSpaceID>::const_iterator it = 
            address_spaces.begin(); it != address_spaces.end(); it++)
        rez.serialize(*it);
    }

    //--------------------------------------------------------------------------
    void ShardMapping::unpack_mapping(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_spaces;
      derez.deserialize(num_spaces);
      address_spaces.resize(num_spaces);
      for (unsigned idx = 0; idx < num_spaces; idx++)
        derez.deserialize(address_spaces[idx]);
    }

    /////////////////////////////////////////////////////////////
    // Shard Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardManager::ShardManager(Runtime *rt, ReplicationID id, bool control, 
                               bool top, size_t total, AddressSpaceID owner, 
                               SingleTask *original/*= NULL*/, RtBarrier bar)
      : runtime(rt), repl_id(id), owner_space(owner), total_shards(total),
        original_task(original),control_replicated(control),
        top_level_task(top), address_spaces(NULL), 
        local_mapping_complete(0), remote_mapping_complete(0),
        local_execution_complete(0), remote_execution_complete(0),
        trigger_local_complete(0), trigger_remote_complete(0),
        trigger_local_commit(0), trigger_remote_commit(0), 
        remote_constituents(0), semantic_attach_counter(0), 
        local_future_result(NULL), local_future_size(0), 
        local_future_set(false), startup_barrier(bar) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(total_shards > 0);
#endif
      // Add an extra reference if we're not the owner manager
      if (owner_space != runtime->address_space)
        add_reference();
      runtime->register_shard_manager(repl_id, this);
      if (control_replicated && (owner_space == runtime->address_space))
      {
#ifdef DEBUG_LEGION
        assert(!startup_barrier.exists());
#endif
        startup_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        pending_partition_barrier = 
          ApBarrier(Realm::Barrier::create_barrier(total_shards));
        // Only need shards-1 for arrivals here since it is used
        // to signal from all the non-creator shards to the creator shard
        creation_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        // Same thing as above for deletion barriers
        deletion_ready_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        deletion_mapping_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        deletion_execution_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        // Inline mapping barrier for synchronizing inline mappings
        // across all the shards
        inline_mapping_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        // External resource barrier for synchronizing attach/detach ops
        external_resource_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        // Fence barriers need arrivals from everyone
        mapping_fence_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        trace_recording_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        summary_fence_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        execution_fence_barrier = 
          ApBarrier(Realm::Barrier::create_barrier(total_shards));
        attach_broadcast_barrier = 
          ApBarrier(Realm::Barrier::create_barrier(1));
        attach_reduce_barrier = 
          ApBarrier(Realm::Barrier::create_barrier(total_shards));
        dependent_partition_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        semantic_attach_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards));
        if (runtime->program_order_execution)
          inorder_barrier = 
            ApBarrier(Realm::Barrier::create_barrier(total_shards));
        // callback barrier can't be made until we know how many
        // unique address spaces we'll actually have so see
        // ShardManager::launch
#ifdef DEBUG_LEGION_COLLECTIVES
        collective_check_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards,
                CollectiveCheckReduction::REDOP,
                &CollectiveCheckReduction::IDENTITY, 
                sizeof(CollectiveCheckReduction::IDENTITY)));
        close_check_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(total_shards,
                CloseCheckReduction::REDOP,
                &CloseCheckReduction::IDENTITY,
                sizeof(CloseCheckReduction::IDENTITY)));
#endif
      }
#ifdef DEBUG_LEGION
      else if (control_replicated)
        assert(startup_barrier.exists());
#endif
    }

    //--------------------------------------------------------------------------
    ShardManager::ShardManager(const ShardManager &rhs)
      : runtime(NULL), repl_id(0), owner_space(0), total_shards(0),
        original_task(NULL), control_replicated(false), top_level_task(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }
    
    //--------------------------------------------------------------------------
    ShardManager::~ShardManager(void)
    //--------------------------------------------------------------------------
    { 
      // We can delete our shard tasks
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
        delete (*it);
      local_shards.clear();
      // Finally unregister ourselves with the runtime
      const bool owner_manager = (owner_space == runtime->address_space);
      runtime->unregister_shard_manager(repl_id, owner_manager);
      if (owner_manager)
      {
        if (control_replicated)
        {
          startup_barrier.destroy_barrier();
          pending_partition_barrier.destroy_barrier();
          creation_barrier.destroy_barrier();
          deletion_ready_barrier.destroy_barrier();
          deletion_mapping_barrier.destroy_barrier();
          deletion_execution_barrier.destroy_barrier();
          inline_mapping_barrier.destroy_barrier();
          external_resource_barrier.destroy_barrier();
          mapping_fence_barrier.destroy_barrier();
          trace_recording_barrier.destroy_barrier();
          summary_fence_barrier.destroy_barrier();
          execution_fence_barrier.destroy_barrier();
          attach_broadcast_barrier.destroy_barrier();
          attach_reduce_barrier.destroy_barrier();
          dependent_partition_barrier.destroy_barrier();
          semantic_attach_barrier.destroy_barrier();
          if (inorder_barrier.exists())
            inorder_barrier.destroy_barrier();
          callback_barrier.destroy_barrier();
#ifdef DEBUG_LEGION_COLLECTIVES
          collective_check_barrier.destroy_barrier();
          close_check_barrier.destroy_barrier();
#endif
        }
        // Send messages to all the remote spaces to remove the manager
        std::set<AddressSpaceID> sent_spaces;
        for (unsigned idx = 0; idx < address_spaces->size(); idx++)
        {
          AddressSpaceID target = (*address_spaces)[idx];
          if (sent_spaces.find(target) != sent_spaces.end())
            continue;
          if (target == runtime->address_space)
            continue;
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(repl_id);
          }
          runtime->send_replicate_delete(target, rez);
          sent_spaces.insert(target);
        }
      }
      if ((address_spaces != NULL) && address_spaces->remove_reference())
        delete address_spaces;
      if (local_future_result != NULL)
        free(local_future_result);
    }

    //--------------------------------------------------------------------------
    ShardManager& ShardManager::operator=(const ShardManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ShardManager::set_shard_mapping(const std::vector<Processor> &mapping)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(mapping.size() == total_shards);
#endif
      shard_mapping = mapping;
    }

    //--------------------------------------------------------------------------
    void ShardManager::set_address_spaces(
                                      const std::vector<AddressSpaceID> &spaces)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(address_spaces == NULL);
#endif
      address_spaces = new ShardMapping(spaces);
      address_spaces->add_reference();
    }

    //--------------------------------------------------------------------------
    void ShardManager::create_callback_barrier(size_t arrival_count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!callback_barrier.exists());
      assert(owner_space == runtime->address_space);
      assert(arrival_count == runtime->total_address_spaces);
#endif
      callback_barrier = 
        RtBarrier(Realm::Barrier::create_barrier(arrival_count));
    }

    //--------------------------------------------------------------------------
    ShardTask* ShardManager::create_shard(ShardID id, Processor target)
    //--------------------------------------------------------------------------
    {
      ShardTask *shard = new ShardTask(runtime, this, id, target);
      local_shards.push_back(shard);
      return shard;
    }

    //--------------------------------------------------------------------------
    void ShardManager::extract_event_preconditions(
                                       const std::deque<InstanceSet> &instances)
    //--------------------------------------------------------------------------
    {
      // Iterate through all the shards and have them extract 
      // their event preconditions
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
        (*it)->extract_event_preconditions(instances);
    }

    //--------------------------------------------------------------------------
    void ShardManager::launch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!local_shards.empty());
      assert(address_spaces == NULL);
#endif
      address_spaces = new ShardMapping();
      address_spaces->add_reference();
      address_spaces->resize(local_shards.size());
      // Sort the shards into their target address space
      std::map<AddressSpaceID,std::vector<ShardTask*> > shard_groups;
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
      {
        const AddressSpaceID target = 
          runtime->find_address_space((*it)->target_proc);
        shard_groups[target].push_back(*it); 
#ifdef DEBUG_LEGION
        assert((*it)->shard_id < address_spaces->size());
#endif
        (*address_spaces)[(*it)->shard_id] = target;
      }
      local_shards.clear();
      // Compute the unique shard spaces and make callback barrier
      // which has as many arrivers as unique shard spaces
      callback_barrier = 
        RtBarrier(Realm::Barrier::create_barrier(shard_groups.size()));
      // Now either send the shards to the remote nodes or record them locally
      for (std::map<AddressSpaceID,std::vector<ShardTask*> >::const_iterator 
            it = shard_groups.begin(); it != shard_groups.end(); it++)
      {
        if (it->first != runtime->address_space)
        {
          distribute_shards(it->first, it->second); 
          // Clean up the shards that are now sent remotely
          for (unsigned idx = 0; idx < it->second.size(); idx++)
            delete it->second[idx];
        }
        else
          local_shards = it->second;
      }
      if (!local_shards.empty())
      {
        for (std::vector<ShardTask*>::const_iterator it = 
              local_shards.begin(); it != local_shards.end(); it++)
          launch_shard(*it);
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::distribute_shards(AddressSpaceID target,
                                         const std::vector<ShardTask*> &shards)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(address_spaces != NULL);
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(repl_id);
        rez.serialize(total_shards);
        rez.serialize(control_replicated);
        rez.serialize(top_level_task);
        rez.serialize(startup_barrier);
        address_spaces->pack_mapping(rez);
        if (control_replicated)
        {
#ifdef DEBUG_LEGION
          assert(pending_partition_barrier.exists());
          assert(creation_barrier.exists());
          assert(deletion_ready_barrier.exists());
          assert(deletion_mapping_barrier.exists());
          assert(deletion_execution_barrier.exists());
          assert(inline_mapping_barrier.exists());
          assert(external_resource_barrier.exists());
          assert(mapping_fence_barrier.exists());
          assert(trace_recording_barrier.exists());
          assert(summary_fence_barrier.exists());
          assert(execution_fence_barrier.exists());
          assert(attach_broadcast_barrier.exists());
          assert(attach_reduce_barrier.exists());
          assert(dependent_partition_barrier.exists());
          assert(semantic_attach_barrier.exists());
          assert(callback_barrier.exists());
          assert(shard_mapping.size() == total_shards);
#endif
          rez.serialize(pending_partition_barrier);
          rez.serialize(creation_barrier);
          rez.serialize(deletion_ready_barrier);
          rez.serialize(deletion_mapping_barrier);
          rez.serialize(deletion_execution_barrier);
          rez.serialize(inline_mapping_barrier);
          rez.serialize(external_resource_barrier);
          rez.serialize(mapping_fence_barrier);
          rez.serialize(trace_recording_barrier);
          rez.serialize(summary_fence_barrier);
          rez.serialize(execution_fence_barrier);
          rez.serialize(attach_broadcast_barrier);
          rez.serialize(attach_reduce_barrier);
          rez.serialize(dependent_partition_barrier);
          rez.serialize(semantic_attach_barrier);
          rez.serialize(inorder_barrier);
          rez.serialize(callback_barrier);
#ifdef DEBUG_LEGION_COLLECTIVES
          assert(collective_check_barrier.exists());
          rez.serialize(collective_check_barrier);
          assert(close_check_barrier.exists());
          rez.serialize(close_check_barrier);
#endif
          for (std::vector<Processor>::const_iterator it = 
                shard_mapping.begin(); it != shard_mapping.end(); it++)
            rez.serialize(*it);
        }
        rez.serialize<size_t>(shards.size());
        for (std::vector<ShardTask*>::const_iterator it = 
              shards.begin(); it != shards.end(); it++)
        {
          rez.serialize((*it)->shard_id);
          rez.serialize((*it)->target_proc);
          (*it)->pack_task(rez, target);
        }
      }
      runtime->send_replicate_launch(target, rez);
      // Update the remote constituents count
      remote_constituents++;
    }

    //--------------------------------------------------------------------------
    void ShardManager::unpack_shards_and_launch(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(owner_space != runtime->address_space);
      assert(local_shards.empty());
      assert(address_spaces == NULL);
#endif
      address_spaces = new ShardMapping();
      address_spaces->add_reference();
      address_spaces->unpack_mapping(derez);
      if (control_replicated)
      {
        derez.deserialize(pending_partition_barrier);
        derez.deserialize(creation_barrier);
        derez.deserialize(deletion_ready_barrier);
        derez.deserialize(deletion_mapping_barrier);
        derez.deserialize(deletion_execution_barrier);
        derez.deserialize(inline_mapping_barrier);
        derez.deserialize(external_resource_barrier);
        derez.deserialize(mapping_fence_barrier);
        derez.deserialize(trace_recording_barrier);
        derez.deserialize(summary_fence_barrier);
        derez.deserialize(execution_fence_barrier);
        derez.deserialize(attach_broadcast_barrier);
        derez.deserialize(attach_reduce_barrier);
        derez.deserialize(dependent_partition_barrier);
        derez.deserialize(semantic_attach_barrier);
        derez.deserialize(inorder_barrier);
        derez.deserialize(callback_barrier);
#ifdef DEBUG_LEGION_COLLECTIVES
        derez.deserialize(collective_check_barrier);
        derez.deserialize(close_check_barrier);
#endif
        shard_mapping.resize(total_shards);
        for (unsigned idx = 0; idx < total_shards; idx++)
          derez.deserialize(shard_mapping[idx]);
      }
      size_t num_shards;
      derez.deserialize(num_shards);
      local_shards.resize(num_shards);
      for (unsigned idx = 0; idx < num_shards; idx++)
      {
        ShardID shard_id;
        derez.deserialize(shard_id);
        Processor target;
        derez.deserialize(target);
        ShardTask *shard = new ShardTask(runtime, this, shard_id, target);
        std::set<RtEvent> ready_preconditions;
        shard->unpack_task(derez, target, ready_preconditions);
        local_shards[idx] = shard;
        if (!ready_preconditions.empty())
          launch_shard(shard, Runtime::merge_events(ready_preconditions));
        else
          launch_shard(shard);
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::launch_shard(ShardTask *task, RtEvent precondition) const
    //--------------------------------------------------------------------------
    {
      ShardManagerLaunchArgs args(task);
      runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, 
                                       precondition);
    }

    //--------------------------------------------------------------------------
    void ShardManager::complete_startup_initialization(void) const
    //--------------------------------------------------------------------------
    {
      // Do our arrival
      Runtime::phase_barrier_arrive(startup_barrier, 1/*count*/);
      // Then wait for everyone else to be ready
      startup_barrier.wait();
    }

    //--------------------------------------------------------------------------
    bool ShardManager::is_total_sharding(void)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(manager_lock);
      if (unique_shard_spaces.empty())
        for (unsigned shard = 0; shard < total_shards; shard++)
              unique_shard_spaces.insert((*address_spaces)[shard]);
      return (unique_shard_spaces.size() == runtime->total_address_spaces);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_post_mapped(bool local, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      bool notify = false;   
      {
        AutoLock m_lock(manager_lock);
        if (precondition.exists())
          mapping_preconditions.insert(precondition);
        if (local)
        {
          local_mapping_complete++;
#ifdef DEBUG_LEGION
          assert(local_mapping_complete <= local_shards.size());
#endif
        }
        else
        {
          remote_mapping_complete++;
#ifdef DEBUG_LEGION
          assert(remote_mapping_complete <= remote_constituents);
#endif
        }
        notify = (local_mapping_complete == local_shards.size()) &&
                 (remote_mapping_complete == remote_constituents);
      }
      if (notify)
      {
        RtEvent mapped_precondition;
        if (!mapping_preconditions.empty())
          mapped_precondition = Runtime::merge_events(mapping_preconditions);
        if (original_task == NULL)
        {
          Serializer rez;
          rez.serialize(repl_id);
          rez.serialize(mapped_precondition);
          runtime->send_replicate_post_mapped(owner_space, rez);
        }
        else
          original_task->handle_post_mapped(false/*deferral*/, 
                                            mapped_precondition);
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_post_execution(const void *res, size_t res_size, 
                                             bool owned, bool local)
    //--------------------------------------------------------------------------
    {
      bool notify = false;
      bool future_claimed = false;
      {
        AutoLock m_lock(manager_lock);
        if (local)
        {
          local_execution_complete++;
#ifdef DEBUG_LEGION
          assert(local_execution_complete <= local_shards.size());
#endif
        }
        else
        {
          remote_execution_complete++;
#ifdef DEBUG_LEGION
          assert(remote_execution_complete <= remote_constituents);
#endif
        }
        notify = (local_execution_complete == local_shards.size()) &&
                 (remote_execution_complete == remote_constituents);
        // See if we need to save the future or compare it
        if (!local_future_set)
        {
          local_future_size = res_size;
          if (!owned)
          {
            local_future_result = malloc(local_future_size);
            memcpy(local_future_result, res, local_future_size);
          }
          else
          {
            local_future_result = const_cast<void*>(res); // take ownership
            future_claimed = true;
          }
          local_future_set = true;
        }
#ifdef DEBUG_LEGION
        // In debug mode we'll do a comparison to see if the futures
        // are bit-wise the same or not and issue a warning if not
        else if ((local_future_size != res_size) || ((local_future_size > 0) && 
                  (strncmp((const char*)res, (const char*)local_future_result, 
                                                      local_future_size) != 0)))
          REPORT_LEGION_WARNING(LEGION_WARNING_MISMATCHED_REPLICATED_FUTURES,
                                "WARNING: futures returned from control "
                                "replicated task %s have different bitwise "
                                "values!", local_shards[0]->get_task_name())
#endif
      }
      if (notify)
      {
        if (original_task == NULL)
        {
          Serializer rez;
          rez.serialize(repl_id);
          rez.serialize<size_t>(local_future_size);
          if (local_future_size > 0)
            rez.serialize(local_future_result, local_future_size);
          runtime->send_replicate_post_execution(owner_space, rez);
        }
        else
        {
          original_task->handle_future(local_future_result, local_future_size, 
                           true/*owned*/, NULL/*functor*/, Processor::NO_PROC);
          local_future_result = NULL;
          local_future_size = 0;
          original_task->complete_execution();
        }
      }
      // if we own it and don't use it we need to free it
      if (owned && !future_claimed)
        free(const_cast<void*>(res));
    }

    //--------------------------------------------------------------------------
    void ShardManager::trigger_task_complete(bool local)
    //--------------------------------------------------------------------------
    {
      bool notify = false;
      {
        AutoLock m_lock(manager_lock);
        if (local)
        {
          trigger_local_complete++;
#ifdef DEBUG_LEGION
          assert(trigger_local_complete <= local_shards.size());
#endif
        }
        else
        {
          trigger_remote_complete++;
#ifdef DEBUG_LEGION
          assert(trigger_remote_complete <= remote_constituents);
#endif
        }
        notify = (trigger_local_complete == local_shards.size()) &&
                 (trigger_remote_complete == remote_constituents);
      }
      if (notify)
      {
        if (original_task == NULL)
        {
          Serializer rez;
          rez.serialize(repl_id);
          runtime->send_replicate_trigger_complete(owner_space, rez);
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(!local_shards.empty());
#endif
          // For one of the shards we either need to return resources up
          // the tree or report leaks and duplicates of resources.
          // All the shards have the same set so we only have to do this
          // for one of the shards.
          std::set<RtEvent> applied;
          if (original_task->is_top_level_task())
            local_shards[0]->report_leaks_and_duplicates(applied);
          else
            local_shards[0]->return_resources(
                original_task->get_context(), applied);
          // We'll just wait for now since there's no good way to
          // force this to be propagated back otherwise
          if (!applied.empty())
          {
            const RtEvent wait_on = Runtime::merge_events(applied);
            if (wait_on.exists() && !wait_on.has_triggered())
              wait_on.wait();
          }
          original_task->trigger_children_complete();
        }
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::trigger_task_commit(bool local)
    //--------------------------------------------------------------------------
    {
      bool notify = false;
      {
        AutoLock m_lock(manager_lock);
        if (local)
        {
          trigger_local_commit++;
#ifdef DEBUG_LEGION
          assert(trigger_local_commit <= local_shards.size());
#endif
        }
        else
        {
          trigger_remote_commit++;
#ifdef DEBUG_LEGION
          assert(trigger_remote_commit <= remote_constituents);
#endif
        }
        notify = (trigger_local_commit == local_shards.size()) &&
                 (trigger_remote_commit == remote_constituents);
      }
      if (notify)
      {
        if (original_task == NULL)
        {
          Serializer rez;
          rez.serialize(repl_id);
          runtime->send_replicate_trigger_commit(owner_space, rez);
        }
        else
          original_task->trigger_children_committed();
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_collective_message(ShardID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target < address_spaces->size());
#endif
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_buffer(), rez.get_used_bytes());
        // Have to unpack the preample we already know
        ReplicationID local_repl;
        derez.deserialize(local_repl);
        handle_collective_message(derez);
      }
      else
        runtime->send_control_replicate_collective_message(target_space, rez);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_collective_message(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
      {
        if ((*it)->shard_id == target)
        {
          (*it)->handle_collective_message(derez);
          return;
        }
      }
      // Should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_future_map_request(ShardID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target < address_spaces->size());
#endif
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_buffer(), rez.get_used_bytes());
        // Have to unpack the preample we already know
        ReplicationID local_repl;
        derez.deserialize(local_repl);     
        handle_future_map_request(derez);
      }
      else
        runtime->send_control_replicate_future_map_request(target_space, rez);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_future_map_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
      {
        if ((*it)->shard_id == target)
        {
          (*it)->handle_future_map_request(derez);
          return;
        }
      }
      // Should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_equivalence_set_request(ShardID target, 
                                                    Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target < address_spaces->size());
#endif
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_buffer(), rez.get_used_bytes());
        // Have to unpack the preample we already know
        ReplicationID local_repl;
        derez.deserialize(local_repl);     
        handle_equivalence_set_request(derez);
      }
      else
        runtime->send_control_replicate_equivalence_set_request(target_space, 
                                                                rez);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_equivalence_set_request(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
      {
        if ((*it)->shard_id == target)
        {
          (*it)->handle_equivalence_set_request(derez);
          return;
        }
      }
      // Should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_intra_space_dependence(ShardID target, 
                                                   Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target < address_spaces->size());
#endif
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_buffer(), rez.get_used_bytes());
        // Have to unpack the preample we already know
        ReplicationID local_repl;
        derez.deserialize(local_repl);     
        handle_intra_space_dependence(derez);
      }
      else
        runtime->send_control_replicate_intra_space_dependence(target_space,
                                                               rez);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_intra_space_dependence(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
      {
        if ((*it)->shard_id == target)
        {
          (*it)->handle_intra_space_dependence(derez);
          return;
        }
      }
      // Should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardManager::broadcast_resource_update(ShardTask *source, 
                             Serializer &rez, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      std::vector<AddressSpaceID> shard_spaces;
      {
        AutoLock m_lock(manager_lock);
        if (unique_shard_spaces.empty())
          for (unsigned shard = 0; shard < total_shards; shard++)
                unique_shard_spaces.insert((*address_spaces)[shard]);
        shard_spaces.insert(shard_spaces.end(), 
            unique_shard_spaces.begin(), unique_shard_spaces.end());
      }
      // First pack it out and send it out to any remote nodes 
      if (shard_spaces.size() > 1)
      {
        // Find the start index
        int start_idx = -1;
        for (unsigned idx = 0; idx < shard_spaces.size(); idx++)
        {
          if (shard_spaces[idx] != runtime->address_space)
            continue;
          start_idx = idx;
          break;
        }
#ifdef DEBUG_LEGION
        assert(start_idx >= 0);
#endif
        std::vector<unsigned> locals;
        std::vector<AddressSpaceID> targets;
        for (int idx = 0; idx < runtime->legion_collective_radix; idx++)
        {
          unsigned next = idx + 1;
          if (next >= shard_spaces.size())
            break;
          locals.push_back(next);
          // Convert from relative to actual address space
          const unsigned next_index = (start_idx + next) % shard_spaces.size();
          targets.push_back(shard_spaces[next_index]);
        }
        for (unsigned idx = 0; idx < locals.size(); idx++)
        {
          RtEvent next_done = Runtime::create_rt_user_event();
          Serializer rez2;
          rez2.serialize(repl_id);
          rez2.serialize<unsigned>(start_idx);
          rez2.serialize<unsigned>(locals[idx]);
          rez2.serialize<size_t>(rez.get_used_bytes());
          rez2.serialize(rez.get_buffer(), rez.get_used_bytes());
          rez2.serialize(next_done);
          runtime->send_control_replicate_resource_update(targets[idx], rez2);
          applied_events.insert(next_done);
        }
      }
      // Then send it to any other local shards
      for (std::vector<ShardTask*>::const_iterator it =
            local_shards.begin(); it != local_shards.end(); it++)
      {
        // Skip the source since that's where it came from
        if ((*it) == source)
          continue;
        Deserializer derez(rez.get_buffer(), rez.get_used_bytes());
        (*it)->handle_resource_update(derez, applied_events);
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_resource_update(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unsigned start_idx, local_idx;
      derez.deserialize(start_idx);
      derez.deserialize(local_idx);
      size_t message_size;
      derez.deserialize(message_size);
      const void *message = derez.get_current_pointer();
      derez.advance_pointer(message_size);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      // Send out any remote updates first
      std::vector<AddressSpaceID> shard_spaces;
      {
        AutoLock m_lock(manager_lock);
        if (unique_shard_spaces.empty())
          for (unsigned shard = 0; shard < total_shards; shard++)
                unique_shard_spaces.insert((*address_spaces)[shard]);
        shard_spaces.insert(shard_spaces.end(), 
            unique_shard_spaces.begin(), unique_shard_spaces.end());
      }
      // First pack it out and send it out to any remote nodes 
      std::vector<unsigned> locals;
      std::vector<AddressSpaceID> targets;
      const unsigned start = local_idx * runtime->legion_collective_radix + 1;
      for (int idx = 0; idx < runtime->legion_collective_radix; idx++)
      {
        unsigned next = start + idx;
        if (next >= shard_spaces.size())
          break;
        locals.push_back(next);
        // Convert from relative to actual address space
        const unsigned next_index = (start_idx + next) % shard_spaces.size();
        targets.push_back(shard_spaces[next_index]);
      }
      std::set<RtEvent> remote_handled;
      if (!targets.empty())
      {
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          RtEvent next_done = Runtime::create_rt_user_event();
          Serializer rez;
          rez.serialize(repl_id);
          rez.serialize<unsigned>(start_idx);
          rez.serialize<unsigned>(locals[idx]);
          rez.serialize<size_t>(message_size);
          rez.serialize(message, message_size);
          rez.serialize(next_done);
          runtime->send_control_replicate_resource_update(targets[idx], rez);
          remote_handled.insert(next_done);
        } 
      }
      // Handle it on all our local shards
      for (std::vector<ShardTask*>::const_iterator it =
            local_shards.begin(); it != local_shards.end(); it++)
      {
        Deserializer derez2(message, message_size);
        (*it)->handle_resource_update(derez2, remote_handled);
      }
      if (!remote_handled.empty())
        Runtime::trigger_event(done_event, 
            Runtime::merge_events(remote_handled));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_trace_event_request(
        ShardedPhysicalTemplate *physical_template, ShardID shard_source, 
        AddressSpaceID template_source, size_t template_index, ApEvent event,
        AddressSpaceID event_space, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      // See whether we are on the right node to handle this request, if not
      // then forward the request onto the proper node
      if (event_space != runtime->address_space)
      {
#ifdef DEBUG_LEGION
        assert(template_source == runtime->address_space);
#endif
        // Check to see if we have a shard on that address space, if not
        // then we know that this event can't have come from there
        bool found = false;
        for (unsigned idx = 0; idx < address_spaces->size(); idx++)
        {
          if ((*address_spaces)[idx] != event_space)
            continue;
          found = true;
          break;
        }
        if (found)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(repl_id);
            rez.serialize(physical_template);
            rez.serialize(template_index);
            rez.serialize(shard_source);
            rez.serialize(event);
            rez.serialize(done_event);
          }
          runtime->send_control_replicate_trace_event_request(event_space, rez);
        }
        else
          send_trace_event_response(physical_template, template_source,
              event, ApBarrier::NO_AP_BARRIER, done_event);
      }
      else
      {
        // Ask each of our local shards to check for the event in the template
        for (std::vector<ShardTask*>::const_iterator it = 
              local_shards.begin(); it != local_shards.end(); it++)
        {
          const ApBarrier result = 
            (*it)->handle_find_trace_shard_event(template_index, 
                                                 event, shard_source);
          // If we found it then we are done
          if (result.exists())
          {
            send_trace_event_response(physical_template, template_source,
                event, result, done_event);
            return;
          }
        }
        // If we make it here then we didn't find it so return the result
        send_trace_event_response(physical_template, template_source,
            event, ApBarrier::NO_AP_BARRIER, done_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_trace_event_request(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardedPhysicalTemplate *physical_template;
      derez.deserialize(physical_template);
      size_t template_index;
      derez.deserialize(template_index);
      ShardID shard_source;
      derez.deserialize(shard_source);
      ApEvent event;
      derez.deserialize(event);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->send_trace_event_request(physical_template, shard_source, source,
          template_index, event, runtime->address_space, done_event);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_trace_event_response(
        ShardedPhysicalTemplate *physical_template, AddressSpaceID temp_source,
        ApEvent event, ApBarrier result, RtUserEvent done_event)
    //--------------------------------------------------------------------------
    {
      if (temp_source != runtime->address_space)
      {
        // Not local so send the response message
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(physical_template);
          rez.serialize(event);
          rez.serialize(result);
          rez.serialize(done_event);
        }
        runtime->send_control_replicate_trace_event_response(temp_source, rez);
      }
      else // This is local so handle it here
      {
        physical_template->record_trace_shard_event(event, result);
        Runtime::trigger_event(done_event);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_trace_event_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ShardedPhysicalTemplate *physical_template;
      derez.deserialize(physical_template);
      ApEvent event;
      derez.deserialize(event);
      ApBarrier result;
      derez.deserialize(result);
      RtUserEvent done_event;
      derez.deserialize(done_event);

      physical_template->record_trace_shard_event(event, result);
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void ShardManager::send_trace_update(ShardID target, Serializer &rez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(target < address_spaces->size());
#endif
      AddressSpaceID target_space = (*address_spaces)[target];
      // Check to see if this is a local shard
      if (target_space == runtime->address_space)
      {
        Deserializer derez(rez.get_buffer(), rez.get_used_bytes());
        // Have to unpack the preample we already know
        ReplicationID local_repl;
        derez.deserialize(local_repl);     
        handle_trace_update(derez, target_space);
      }
      else
        runtime->send_control_replicate_trace_update(target_space, rez);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_trace_update(Deserializer &derez,
                                           AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // Figure out which shard we are going to
      ShardID target;
      derez.deserialize(target);
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
      {
        if ((*it)->shard_id == target)
        {
          (*it)->handle_trace_update(derez, source);
          return;
        }
      }
      // Should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_launch(const void *args)
    //--------------------------------------------------------------------------
    {
      const ShardManagerLaunchArgs *largs = (const ShardManagerLaunchArgs*)args;
      largs->shard->launch_shard();
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_delete(const void *args)
    //--------------------------------------------------------------------------
    {
      const ShardManagerDeleteArgs *dargs = (const ShardManagerDeleteArgs*)args;
      if (dargs->manager->remove_reference())
        delete dargs->manager;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_launch(Deserializer &derez, 
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      size_t total_shards;
      derez.deserialize(total_shards);
      bool control_repl;
      derez.deserialize(control_repl);
      bool top_level_task;
      derez.deserialize(top_level_task);
      RtBarrier startup_barrier;
      derez.deserialize(startup_barrier);
      ShardManager *manager = 
        new ShardManager(runtime, repl_id, control_repl, top_level_task,
                total_shards, source, NULL/*original*/, startup_barrier);
      manager->unpack_shards_and_launch(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_delete(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      delete manager;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_post_mapped(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      RtEvent precondition;
      derez.deserialize(precondition);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_post_mapped(false/*local*/, precondition);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_post_execution(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      size_t future_result_size;
      derez.deserialize(future_result_size);
      const void *future_result = derez.get_current_pointer();
      if (future_result_size > 0)
        derez.advance_pointer(future_result_size);
      manager->handle_post_execution(future_result, future_result_size,
                                     false/*owned*/, false/*local*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_trigger_complete(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->trigger_task_complete(false/*local*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_trigger_commit(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->trigger_task_commit(false/*local*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_collective_message(Deserializer &derez,
                                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_collective_message(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_future_map_request(Deserializer &derez,
                                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_future_map_request(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_trace_update(Deserializer &derez,
                                                      Runtime *runtime,
                                                      AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_trace_update(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_top_view_request(Deserializer &derez,
                                Runtime *runtime, AddressSpaceID request_source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      DistributedID manager_did;
      derez.deserialize(manager_did);
      AddressSpaceID source;
      derez.deserialize(source);
      ReplicateContext *request_context;
      derez.deserialize(request_context);

      RtEvent ready;
      PhysicalManager *physical_manager = 
        runtime->find_or_request_instance_manager(manager_did, ready); 
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      if (!ready.has_triggered())
        ready.wait();
      manager->create_instance_top_view(physical_manager, source, 
                request_context, request_source, true/*handle now*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_top_view_response(Deserializer &derez,
                                                           Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DistributedID manager_did, view_did;
      derez.deserialize(manager_did);
      derez.deserialize(view_did);
      ReplicateContext *request_context;
      derez.deserialize(request_context);

      RtEvent manager_ready, view_ready;
      PhysicalManager *manager = 
        runtime->find_or_request_instance_manager(manager_did, manager_ready);
      InstanceView *view = static_cast<InstanceView*>(
          runtime->find_or_request_logical_view(view_did, view_ready));
      if (!manager_ready.has_triggered())
        manager_ready.wait();
      if (!view_ready.has_triggered())
        view_ready.wait();
      request_context->record_replicate_instance_top_view(manager, view);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_eq_request(Deserializer &derez,
                                                    Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_equivalence_set_request(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_intra_space_dependence(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_intra_space_dependence(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_resource_update(Deserializer &derez,
                                                         Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_resource_update(derez);
    }

    //--------------------------------------------------------------------------
    ShardingFunction* ShardManager::find_sharding_function(ShardingID sid)
    //--------------------------------------------------------------------------
    {
      // Check to see if it is in the cache
      {
        AutoLock m_lock(manager_lock,1,false/*exclusive*/);
        std::map<ShardingID,ShardingFunction*>::const_iterator finder = 
          sharding_functions.find(sid);
        if (finder != sharding_functions.end())
          return finder->second;
      }
      // Get the functor from the runtime
      ShardingFunctor *functor = runtime->find_sharding_functor(sid);
      // Retake the lock
      AutoLock m_lock(manager_lock);
      // See if we lost the race
      std::map<ShardingID,ShardingFunction*>::const_iterator finder = 
        sharding_functions.find(sid);
      if (finder != sharding_functions.end())
        return finder->second;
      ShardingFunction *result = 
        new ShardingFunction(functor, runtime->forest, sid, total_shards);
      // Save the result for the future
      sharding_functions[sid] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void ShardManager::create_instance_top_view(PhysicalManager *manager, 
                AddressSpaceID source, ReplicateContext *request_context, 
                AddressSpaceID request_source, bool handle_now/*= false*/)
    //--------------------------------------------------------------------------
    {
      // Easy case if we are not control replicated
      if (!control_replicated)
      {
        InstanceView *result = 
          request_context->create_replicate_instance_top_view(manager, source);
        request_context->record_replicate_instance_top_view(manager, result);
        return;
      }
      // If we're on the owner node of the manager just handle it here
      if (handle_now || (manager->owner_space == runtime->address_space))
      {
#ifdef DEBUG_LEGION
        assert(!local_shards.empty());
#endif
        // Distribute manager requests across local shards
        const unsigned index = manager->did % local_shards.size();
        InstanceView *result = 
          local_shards[index]->create_instance_top_view(manager, source);
        // Now we have to tell the request context about the result
        if (request_source != runtime->address_space)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(manager->did);
            rez.serialize(result->did);
            rez.serialize(request_context);
          }
          runtime->send_control_replicate_top_view_response(request_source,rez);
        }
        else
          request_context->record_replicate_instance_top_view(manager, result);
      }
      else
      {
        // Check to see if we already have a manager on the owner node
        // if so we can just send a message there and handle it
        // If not, we round robin the distributed ID across the shards to
        // find the shard to handle the request and send it there
        AddressSpaceID target;
        {
          AutoLock m_lock(manager_lock);
          if (unique_shard_spaces.empty())
            for (unsigned shard = 0; shard < total_shards; shard++)
              unique_shard_spaces.insert((*address_spaces)[shard]);
          if (unique_shard_spaces.find(manager->owner_space) == 
              unique_shard_spaces.end())
          {
            // Round-robin accross the shards
            const unsigned index = manager->did % total_shards;
            target = (*address_spaces)[index];
          }
          else
            target = manager->owner_space;
        }
        if (target != runtime->address_space)
        {
          // Now we can send the message to the target
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(repl_id);
            rez.serialize(manager->did);
            rez.serialize(source);
            rez.serialize(request_context);
          }
          runtime->send_control_replicate_top_view_request(target, rez);
        }
        else
          create_instance_top_view(manager, source, request_context, 
                                   request_source, true/*handle now*/);
      }
    }

#ifdef LEGION_USE_LIBDL
    //--------------------------------------------------------------------------
    void ShardManager::perform_global_registration_callbacks(
                     Realm::DSOReferenceImplementation *dso, RtEvent local_done,
                     RtEvent global_done, std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // See if we're the first one to handle this DSO
      const std::pair<std::string,std::string> 
        key(dso->dso_name, dso->symbol_name);
      {
        AutoLock m_lock(manager_lock);
        // Check to see if we've already handled this
        std::set<std::pair<std::string,std::string> >::const_iterator finder =
          unique_registration_callbacks.find(key);
        if (finder != unique_registration_callbacks.end())
          return;
        unique_registration_callbacks.insert(key);
        if (unique_shard_spaces.empty())
          for (unsigned shard = 0; shard < total_shards; shard++)
                unique_shard_spaces.insert((*address_spaces)[shard]);
      }
      // We're the first one so handle it
      if (!is_total_sharding())
      {
        std::set<RtEvent> local_preconditions;
        AddressSpaceID space = 0;
        for (std::set<AddressSpaceID>::const_iterator it = 
              unique_shard_spaces.begin(); it != 
              unique_shard_spaces.end(); it++, space++)
        {
          if ((*it) == runtime->address_space)
            break;
        }
#ifdef DEBUG_LEGION
        assert(space < unique_shard_spaces.size());
#endif
        for ( ; space < runtime->total_address_spaces; 
              space += unique_shard_spaces.size())
        {
          if (unique_shard_spaces.find(space) != unique_shard_spaces.end())
            continue;
          runtime->send_registration_callback(space, dso, global_done, 
                                              local_preconditions);
        }
        if (!local_preconditions.empty())
        {
          local_preconditions.insert(local_done);
          Runtime::phase_barrier_arrive(callback_barrier, 1/*count*/,
                          Runtime::merge_events(local_preconditions));
        }
        else
          Runtime::phase_barrier_arrive(callback_barrier,
                                        1/*count*/, local_done);
      }
      else // there will be a callback on every node anyway
        Runtime::phase_barrier_arrive(callback_barrier,1/*count*/,local_done);
      preconditions.insert(callback_barrier);
      Runtime::advance_barrier(callback_barrier);
      if (!callback_barrier.exists())
        REPORT_LEGION_FATAL(LEGION_FATAL_UNIMPLEMENTED_FEATURE,
            "Need support for refreshing exhausted callback phase "
            "barrier generations.")
    }
#endif // LEGION_USE_LIBDL

    //--------------------------------------------------------------------------
    bool ShardManager::perform_semantic_attach(void)
    //--------------------------------------------------------------------------
    {
      if (local_shards.size() == 1)
        return true;
      AutoLock m_lock(manager_lock);
#ifdef DEBUG_LEGION
      assert(semantic_attach_counter < local_shards.size());
#endif
      if (++semantic_attach_counter == local_shards.size())
      {
        semantic_attach_counter = 0;
        return true;
      }
      else
        return false;
    }

    /////////////////////////////////////////////////////////////
    // Shard Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardCollective::ShardCollective(CollectiveIndexLocation loc,
                                     ReplicateContext *ctx)
      : manager(ctx->shard_manager), context(ctx), 
        local_shard(ctx->owner_shard->shard_id), 
        collective_index(ctx->get_next_collective_index(loc))
    //--------------------------------------------------------------------------
    {
      context->add_reference();
    }

    //--------------------------------------------------------------------------
    ShardCollective::ShardCollective(ReplicateContext *ctx, CollectiveID id)
      : manager(ctx->shard_manager), context(ctx), 
        local_shard(ctx->owner_shard->shard_id), collective_index(id)
    //--------------------------------------------------------------------------
    { 
      context->add_reference();
    }

    //--------------------------------------------------------------------------
    ShardCollective::~ShardCollective(void)
    //--------------------------------------------------------------------------
    {
      // Unregister this with the context 
      context->unregister_collective(this);
      if (context->remove_reference())
        delete context;
    }

    //--------------------------------------------------------------------------
    int ShardCollective::convert_to_index(ShardID id, ShardID origin) const
    //--------------------------------------------------------------------------
    {
      // shift everything so that the target shard is at index 0
      const int result = 
        ((id + (manager->total_shards - origin)) % manager->total_shards);
      return result;
    }

    //--------------------------------------------------------------------------
    ShardID ShardCollective::convert_to_shard(int index, ShardID origin) const
    //--------------------------------------------------------------------------
    {
      // Add target then take the modulus
      const ShardID result = (index + origin) % manager->total_shards; 
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Gather Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BroadcastCollective::BroadcastCollective(CollectiveIndexLocation loc,
                                             ReplicateContext *ctx, ShardID o)
      : ShardCollective(loc, ctx), origin(o),
        shard_collective_radix(ctx->get_shard_collective_radix())
    //--------------------------------------------------------------------------
    {
      if (local_shard != origin)
        done_event = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    BroadcastCollective::BroadcastCollective(ReplicateContext *ctx, 
                                             CollectiveID id, ShardID o)
      : ShardCollective(ctx, id), origin(o),
        shard_collective_radix(ctx->get_shard_collective_radix())
    //--------------------------------------------------------------------------
    {
      if (local_shard != origin)
        done_event = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    BroadcastCollective::~BroadcastCollective(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void BroadcastCollective::perform_collective_async(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_shard == origin);
#endif
      // Register this with the context
      context->register_collective(this);
      send_messages(); 
    }

    //--------------------------------------------------------------------------
    RtEvent BroadcastCollective::perform_collective_wait(bool block/*=true*/)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_shard != origin);
#endif     
      // Register this with the context
      context->register_collective(this);
      if (!done_event.has_triggered())
      {
        if (block)
          done_event.wait();
        else
          return done_event;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void BroadcastCollective::handle_collective_message(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_shard != origin);
#endif
      // No need for the lock since this is only written to once
      unpack_collective(derez);
      // Send our messages
      send_messages();
      // Then trigger our event to indicate that we are ready
      Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    RtEvent BroadcastCollective::get_done_event(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_shard != origin);
#endif
      return done_event;
    }

    //--------------------------------------------------------------------------
    void BroadcastCollective::send_messages(void) const
    //--------------------------------------------------------------------------
    {
      const int local_index = convert_to_index(local_shard, origin);
      for (int idx = 1; idx <= shard_collective_radix; idx++)
      {
        const int target_index = local_index * shard_collective_radix + idx; 
        if (target_index >= int(manager->total_shards))
          break;
        ShardID target = convert_to_shard(target_index, origin);
        Serializer rez;
        {
          rez.serialize(manager->repl_id);
          rez.serialize(target);
          rez.serialize(collective_index);
          pack_collective(rez);
        }
        manager->send_collective_message(target, rez);
      }
    }

    /////////////////////////////////////////////////////////////
    // Gather Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    GatherCollective::GatherCollective(CollectiveIndexLocation loc,
                                       ReplicateContext *ctx, ShardID t)
      : ShardCollective(loc, ctx), target(t), 
        shard_collective_radix(ctx->get_shard_collective_radix()),
        expected_notifications(compute_expected_notifications()),
        received_notifications(0)
    //--------------------------------------------------------------------------
    {
      if (expected_notifications > 1)
        done_event = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    GatherCollective::~GatherCollective(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (done_event.exists())
        assert(done_event.has_triggered());
#endif
    }

    //--------------------------------------------------------------------------
    void GatherCollective::perform_collective_async(void)
    //--------------------------------------------------------------------------
    {
      // Register this with the context
      context->register_collective(this);
      bool done = false;
      {
        AutoLock c_lock(collective_lock);
#ifdef DEBUG_LEGION
        assert(received_notifications < expected_notifications);
#endif
        done = (++received_notifications == expected_notifications);
      }
      if (done)
      {
        if (local_shard != target)
          send_message();
        if (done_event.exists())
          Runtime::trigger_event(done_event);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent GatherCollective::perform_collective_wait(bool block/*=true*/)
    //--------------------------------------------------------------------------
    {
      if (done_event.exists() && !done_event.has_triggered())
      {
        if (block)
          done_event.wait();
        else
          return done_event;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void GatherCollective::handle_collective_message(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      {
        // Hold the lock while doing these operations
        AutoLock c_lock(collective_lock);
        // Unpack the result
        unpack_collective(derez);
 #ifdef DEBUG_LEGION
        assert(received_notifications < expected_notifications);
#endif
        done = (++received_notifications == expected_notifications);       
      }
      if (done)
      {
        if (local_shard != target)
          send_message();
        if (done_event.exists())
          Runtime::trigger_event(done_event);
      }
    }

    //--------------------------------------------------------------------------
    void GatherCollective::elide_collective(void)
    //--------------------------------------------------------------------------
    {
      if (done_event.exists())
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    void GatherCollective::send_message(void)
    //--------------------------------------------------------------------------
    {
      // Convert to our local index
      const int local_index = convert_to_index(local_shard, target);
#ifdef DEBUG_LEGION
      assert(local_index > 0); // should never be here for zero
#endif
      // Subtract by 1 and then divide to get the target (truncate)
      const int target_index = (local_index - 1) / shard_collective_radix;
      // Then convert back to the target
      ShardID next = convert_to_shard(target_index, target);
      Serializer rez;
      {
        rez.serialize(manager->repl_id);
        rez.serialize(next);
        rez.serialize(collective_index);
        AutoLock c_lock(collective_lock,1,false/*exclusive*/);
        pack_collective(rez);
      }
      manager->send_collective_message(next, rez);
    } 

    //--------------------------------------------------------------------------
    int GatherCollective::compute_expected_notifications(void) const
    //--------------------------------------------------------------------------
    {
      int result = 1; // always have one arriver for ourself
      const int index = convert_to_index(local_shard, target);
      for (int idx = 1; idx <= shard_collective_radix; idx++)
      {
        const int source_index = index * shard_collective_radix + idx;
        if (source_index >= int(manager->total_shards))
          break;
        result++;
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // All Gather Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<bool INORDER>
    AllGatherCollective<INORDER>::AllGatherCollective(
        CollectiveIndexLocation loc, ReplicateContext *ctx)
      : ShardCollective(loc, ctx),
        shard_collective_radix(ctx->get_shard_collective_radix()),
        shard_collective_log_radix(ctx->get_shard_collective_log_radix()),
        shard_collective_stages(ctx->get_shard_collective_stages()),
        shard_collective_participating_shards(
            ctx->get_shard_collective_participating_shards()),
        shard_collective_last_radix(ctx->get_shard_collective_last_radix()),
        participating(int(local_shard) < shard_collective_participating_shards),
        reorder_stages(NULL), pending_send_ready_stages(0)
#ifdef DEBUG_LEGION
        , done_triggered(false)
#endif
    //--------------------------------------------------------------------------
    { 
      initialize_collective(); 
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    AllGatherCollective<INORDER>::AllGatherCollective(ReplicateContext *ctx,
                                                      CollectiveID id)
      : ShardCollective(ctx, id),
        shard_collective_radix(ctx->get_shard_collective_radix()),
        shard_collective_log_radix(ctx->get_shard_collective_log_radix()),
        shard_collective_stages(ctx->get_shard_collective_stages()),
        shard_collective_participating_shards(
            ctx->get_shard_collective_participating_shards()),
        shard_collective_last_radix(ctx->get_shard_collective_last_radix()),
        participating(int(local_shard) < shard_collective_participating_shards),
        reorder_stages(NULL), pending_send_ready_stages(0)
#ifdef DEBUG_LEGION
        , done_triggered(false)
#endif
    //--------------------------------------------------------------------------
    {
      initialize_collective();
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::initialize_collective(void)
    //--------------------------------------------------------------------------
    {
      if (manager->total_shards > 1)
      {
        // We already have our contributions for each stage so
        // we can set the inditial participants to 1
        if (participating)
        {
#ifdef DEBUG_LEGION
          assert(shard_collective_stages > 0);
#endif
          sent_stages.resize(shard_collective_stages, false);
          stage_notifications.resize(shard_collective_stages, 1);
          // Stage 0 always starts with 0 notifications since we'll 
          // explictcly arrive on it
          stage_notifications[0] = 0;
        }
        done_event = Runtime::create_rt_user_event();
      }
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    AllGatherCollective<INORDER>::~AllGatherCollective(void)
    //--------------------------------------------------------------------------
    {
      if (reorder_stages != NULL)
      {
#ifdef DEBUG_LEGION
        assert(reorder_stages->empty());
#endif
        delete reorder_stages;
      }
#ifdef DEBUG_LEGION
      if (participating)
      {
        // We should have sent all our stages before being deleted
        for (unsigned idx = 0; idx < sent_stages.size(); idx++)
          assert(sent_stages[idx]);
      }
      if (participating)
        assert(done_triggered);
      assert(done_event.has_triggered());
#endif
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::perform_collective_sync(void)
    //--------------------------------------------------------------------------
    {
      perform_collective_async(); 
      perform_collective_wait();
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::perform_collective_async(void)
    //--------------------------------------------------------------------------
    {
      // Register this with the context
      context->register_collective(this);
      if (manager->total_shards <= 1)
        return;
      // See if we are a participating shard or not
      if (participating)
      {
        // We are a participating shard 
        // See if we are waiting for an initial notification
        // if not we can just send our message now
        if ((int(manager->total_shards) == 
              shard_collective_participating_shards) || 
            (local_shard >= (manager->total_shards -
              shard_collective_participating_shards)))
        {
          const bool all_stages_done = initiate_collective();
          if (all_stages_done)
            complete_exchange();
        }
      }
      else
      {
        // We are not a participating shard
        // so we just have to send notification to one shard
        send_remainder_stage();
      }
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    RtEvent AllGatherCollective<INORDER>::perform_collective_wait(
                                                            bool block/*=true*/)
    //--------------------------------------------------------------------------
    {
      if (manager->total_shards <= 1)
        return RtEvent::NO_RT_EVENT;
      if (!done_event.has_triggered())
      {
        if (block)
          done_event.wait();
        else
          return done_event;
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::handle_collective_message(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      int stage;
      derez.deserialize(stage);
#ifdef DEBUG_LEGION
      assert(participating || (stage == -1));
#endif
      unpack_stage(stage, derez);
      bool all_stages_done = false;
      if (stage == -1)
      {
        if (!participating)
          all_stages_done = true;
        else // we can now initiate the collective
          all_stages_done = initiate_collective(); 
      }
      else
        all_stages_done = send_ready_stages();
      if (all_stages_done)
        complete_exchange();
    } 

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::elide_collective(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // make it look like we sent all the stages
      for (unsigned idx = 0; idx < sent_stages.size(); idx++)
        sent_stages[idx] = true;
      assert(!done_triggered);
      assert(!done_event.has_triggered());
#endif
      // Trigger the user event 
      Runtime::trigger_event(done_event);
#ifdef DEBUG_LEGION
      done_triggered = true;
#endif
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::construct_message(ShardID target, 
                                                     int stage, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      rez.serialize(manager->repl_id);
      rez.serialize(target);
      rez.serialize(collective_index);
      rez.serialize(stage);
      AutoLock c_lock(collective_lock, 1, false/*exclusive*/);
      pack_collective_stage(rez, stage);
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    bool AllGatherCollective<INORDER>::initiate_collective(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(participating); // should only get this for participating shards
#endif
      {
        AutoLock c_lock(collective_lock);
#ifdef DEBUG_LEGION
        assert(!sent_stages.empty());
        assert(!sent_stages[0]); // stage 0 shouldn't be sent yet
        assert(!stage_notifications.empty());
        if (shard_collective_stages == 1)
          assert(stage_notifications[0] < shard_collective_last_radix); 
        else
          assert(stage_notifications[0] < shard_collective_radix);
#endif
        stage_notifications[0]++;
        // Increment our guard to prevent deletion of the collective
        // object while we are still traversing
        pending_send_ready_stages++;
      }
      return send_ready_stages(0/*start stage*/);
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::send_remainder_stage(void)
    //--------------------------------------------------------------------------
    {
      if (participating)
      {
        // Send back to the shards that are not participating
        ShardID target = local_shard + shard_collective_participating_shards;
#ifdef DEBUG_LEGION
        assert(target < manager->total_shards);
#endif
        Serializer rez;
        construct_message(target, -1/*stage*/, rez);
        manager->send_collective_message(target, rez);
      }
      else
      {
        // Send to a node that is participating
        ShardID target = local_shard % shard_collective_participating_shards;
        Serializer rez;
        construct_message(target, -1/*stage*/, rez);
        manager->send_collective_message(target, rez);
      }
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    bool AllGatherCollective<INORDER>::send_ready_stages(const int start_stage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(participating);
#endif
      // Iterate through the stages and send any that are ready
      // Remember that stages have to be done in order
      bool sent_previous_stage = false;
      for (int stage = start_stage; stage < shard_collective_stages; stage++)
      {
        {
          AutoLock c_lock(collective_lock);
          if (sent_previous_stage)
          {
#ifdef DEBUG_LEGION
            assert(!sent_stages[stage-1]);
#endif
            sent_stages[stage-1] = true;
            sent_previous_stage = false;
          }
          // If this stage has already been sent then we can keep going
          if (sent_stages[stage])
            continue;
#ifdef DEBUG_LEGION
          assert(pending_send_ready_stages > 0);
#endif
          // Check to see if we're sending this stage
          // We need all the notifications from the previous stage before
          // we can send this stage
          if (stage > 0)
          {
            // We can't have multiple threads doing sends at the same time
            // so make sure that only the last one is going through doing work
            // but stage 0 is because it is always sent by the initiator so
            // don't check this until we're past the first stage
            if ((stage_notifications[stage-1] < shard_collective_radix) ||
                (pending_send_ready_stages > 1))
            {
              // Remove our guard before exiting early
              pending_send_ready_stages--;
              return false;
            }
            else if (INORDER && (reorder_stages != NULL))
            {
              // Check to see if we have any unhandled messages for 
              // the previous stage that we need to handle before sending
              std::map<int,std::vector<std::pair<void*,size_t> > >::iterator
                finder = reorder_stages->find(stage-1);
              if (finder != reorder_stages->end())
              {
                // Perform the handling for the buffered messages now
                for (std::vector<std::pair<void*,size_t> >::const_iterator it =
                      finder->second.begin(); it != finder->second.end(); it++)
                {
                  Deserializer derez(it->first, it->second);
                  unpack_collective_stage(derez, finder->first);
                  free(it->first);
                }
                reorder_stages->erase(finder);
              }
            }
          }
          // If we get here then we can send the stage
        }
        // Now we can do the send
        if (stage == (shard_collective_stages-1))
        {
          for (int r = 1; r < shard_collective_last_radix; r++)
          {
            const ShardID target = local_shard ^
              (r << (stage * shard_collective_log_radix));
#ifdef DEBUG_LEGION
            assert(int(target) < shard_collective_participating_shards);
#endif
            Serializer rez;
            construct_message(target, stage, rez);
            manager->send_collective_message(target, rez);
          }
        }
        else
        {
          for (int r = 1; r < shard_collective_radix; r++)
          {
            const ShardID target = local_shard ^
              (r << (stage * shard_collective_log_radix));
#ifdef DEBUG_LEGION
            assert(int(target) < shard_collective_participating_shards);
#endif
            Serializer rez;
            construct_message(target, stage, rez);
            manager->send_collective_message(target, rez);
          }
        }
        sent_previous_stage = true;
      }
      // If we make it here, then we sent the last stage, check to see
      // if we've seen all the notifications for it
      AutoLock c_lock(collective_lock);
      if (sent_previous_stage)
      {
#ifdef DEBUG_LEGION
        assert(!sent_stages[shard_collective_stages-1]);
#endif
        sent_stages[shard_collective_stages-1] = true;
      }
      // Remove our pending guard and then check to see if we are done
#ifdef DEBUG_LEGION
      assert(pending_send_ready_stages > 0);
#endif
      if (((--pending_send_ready_stages) == 0) &&
          (stage_notifications.back() == shard_collective_last_radix))
      {
#ifdef DEBUG_LEGION
        assert(!done_triggered);
        done_triggered = true;
#endif
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::unpack_stage(int stage, 
                                                    Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(collective_lock);
      // Do the unpack first while holding the lock
      if (INORDER && (stage >= 0))
      {
        // Check to see if we can handle this message now or whether we
        // need to buffer it for the future because we have not finished
        // sending the current stage yet or not
        if (!sent_stages[stage])
        {
          // Buffer this message until the stage is sent as well 
          const size_t buffer_size = derez.get_remaining_bytes();
          void *buffer = malloc(buffer_size);
          memcpy(buffer, derez.get_current_pointer(), buffer_size);
          derez.advance_pointer(buffer_size);
          if (reorder_stages == NULL)
            reorder_stages = 
              new std::map<int,std::vector<std::pair<void*,size_t> > >();
          (*reorder_stages)[stage].push_back(
              std::pair<void*,size_t>(buffer, buffer_size));
        }
        else
          unpack_collective_stage(derez, stage);
      }
      else // Just do the unpack here immediately
        unpack_collective_stage(derez, stage);
      if (stage >= 0)
      {
#ifdef DEBUG_LEGION
	assert(stage < int(stage_notifications.size()));
        if (stage < (shard_collective_stages-1))
          assert(stage_notifications[stage] < shard_collective_radix);
        else
          assert(stage_notifications[stage] < shard_collective_last_radix);
#endif
        stage_notifications[stage]++;
        // Increment our guard to prevent deletion of the collective
        // object while we are still traversing
        pending_send_ready_stages++;
      }
    }

    //--------------------------------------------------------------------------
    template<bool INORDER>
    void AllGatherCollective<INORDER>::complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      if ((reorder_stages != NULL) && !reorder_stages->empty())
      {
#ifdef DEBUG_LEGION
        assert(reorder_stages->size() == 1);
#endif
        std::map<int,std::vector<std::pair<void*,size_t> > >::iterator 
          remaining = reorder_stages->begin();
        for (std::vector<std::pair<void*,size_t> >::const_iterator it = 
              remaining->second.begin(); it != remaining->second.end(); it++)
        {
          Deserializer derez(it->first, it->second);
          unpack_collective_stage(derez, remaining->first);
          free(it->first);     
        }
        reorder_stages->erase(remaining);
      }
      // See if we have to send a message back to a non-participating shard 
      if ((int(manager->total_shards) > shard_collective_participating_shards)
          && (int(local_shard) < int(manager->total_shards -
                                     shard_collective_participating_shards)))
        send_remainder_stage();
      // Only after we send this message can we mark that we're done
      Runtime::trigger_event(done_event);
    }

    /////////////////////////////////////////////////////////////
    // All Reduce Op Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AllReduceOpCollective::AllReduceOpCollective(CollectiveIndexLocation loc,
                                  ReplicateContext *ctx, const ReductionOp *op)
      : AllGatherCollective(loc, ctx), redop(op), current_stage(-1),
        value(malloc(op->sizeof_rhs))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AllReduceOpCollective::AllReduceOpCollective(ReplicateContext *ctx,
                                         CollectiveID id, const ReductionOp* op)
      : AllGatherCollective(ctx, id), redop(op), current_stage(-1),
        value(malloc(op->sizeof_rhs))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    AllReduceOpCollective::~AllReduceOpCollective(void)
    //--------------------------------------------------------------------------
    {
      while (!future_values.empty())
      {
        std::map<int,std::vector<void*> >::iterator next = 
          future_values.begin();
        for (std::vector<void*>::iterator it = 
              next->second.begin(); it != next->second.end(); it++)
          free(*it);
        future_values.erase(next);
      }
      free(value);
    }

    //--------------------------------------------------------------------------
    void AllReduceOpCollective::pack_collective_stage(Serializer &rez,int stage)
    //--------------------------------------------------------------------------
    {
      // The first time we pack a stage we merge any values that we had
      // unpacked earlier as they are needed for sending this stage for
      // the first time.
      if (stage != current_stage)
      {
        if (!future_values.empty())
        {
          std::map<int,std::vector<void*> >::iterator next = 
            future_values.begin();
          if (next->first == current_stage)
          {
            for (std::vector<void*>::const_iterator it = 
                  next->second.begin(); it != next->second.end(); it++)
            {
              redop->fold(value, *it, 1/*count*/, true/*exclusive*/);
              free(*it);
            }
            future_values.erase(next);
          }
        }
        current_stage = stage;
      }
      rez.serialize(value, redop->sizeof_rhs);
    }

    //--------------------------------------------------------------------------
    void AllReduceOpCollective::unpack_collective_stage(
                                                 Deserializer &derez, int stage)
    //--------------------------------------------------------------------------
    {
      // We never eagerly do reductions as they can arrive out of order
      // and we can't apply them too early or we'll get duplicate 
      // applications of reductions
      void *next = malloc(redop->sizeof_rhs);
      derez.deserialize(next, redop->sizeof_rhs);
      future_values[stage].push_back(next);
    }

    //--------------------------------------------------------------------------
    RtEvent AllReduceOpCollective::async_reduce(const void *input)
    //--------------------------------------------------------------------------
    {
      memcpy(value, input, redop->sizeof_rhs);
      perform_collective_async();
      return perform_collective_wait(false/*block*/);
    }
    
    //--------------------------------------------------------------------------
    void AllReduceOpCollective::sync_result(void *result)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait(true/*block*/); 
      // Need to avoid races here so we have to always recompute the last stage
      memcpy(result, value, redop->sizeof_rhs);
      if (!future_values.empty())
      {
#ifdef DEBUG_LEGION
        // Should be at most one stage left
        assert(future_values.size() == 1);
#endif
        const std::map<int,std::vector<void*> >::const_iterator last = 
          future_values.begin();
        if (last->first == -1)
        {
          // Special case for the last stage which already includes our
          // value so just do the overwrite
#ifdef DEBUG_LEGION
          assert(last->second.size() == 1);
#endif
          memcpy(result, last->second.front(), redop->sizeof_rhs);
        }
        else
        {
          // Do the reduction here
          for (std::vector<void*>::const_iterator it =
                last->second.begin(); it != last->second.end(); it++)
            redop->fold(result, *it, 1/*count*/, true/*exclusive*/);
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // All Reduce Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename REDOP>
    AllReduceCollective<REDOP>::AllReduceCollective(CollectiveIndexLocation loc,
                                                    ReplicateContext *ctx)
      : AllGatherCollective(loc, ctx), current_stage(-1)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    AllReduceCollective<REDOP>::AllReduceCollective(ReplicateContext *ctx,
                                                    CollectiveID id)
      : AllGatherCollective(ctx, id), current_stage(-1)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    AllReduceCollective<REDOP>::~AllReduceCollective(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    void AllReduceCollective<REDOP>::pack_collective_stage(Serializer &rez,
                                                           int stage)
    //--------------------------------------------------------------------------
    {
      // The first time we pack a stage we merge any values that we had
      // unpacked earlier as they are needed for sending this stage for
      // the first time.
      if (stage != current_stage)
      {
        if (!future_values.empty())
        {
          typename std::map<int,std::vector<typename REDOP::RHS> >::iterator 
            next = future_values.begin();
          if (next->first == current_stage)
          {
            for (typename std::vector<typename REDOP::RHS>::const_iterator it =
                  next->second.begin(); it != next->second.end(); it++)
              REDOP::template fold<true/*exclusive*/>(value, *it);
            future_values.erase(next);
          }
        }
        current_stage = stage;
      }
      rez.serialize(value);
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    void AllReduceCollective<REDOP>::unpack_collective_stage(
                                                 Deserializer &derez, int stage)
    //--------------------------------------------------------------------------
    {
      // We never eagerly do reductions as they can arrive out of order
      // and we can't apply them too early or we'll get duplicate 
      // applications of reductions
      typename REDOP::RHS next;
      derez.deserialize(next);
      future_values[stage].push_back(next);
    }
    
    //--------------------------------------------------------------------------
    template<typename REDOP>
    void AllReduceCollective<REDOP>::async_all_reduce(typename REDOP::RHS val)
    //--------------------------------------------------------------------------
    {
      value = val;
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    RtEvent AllReduceCollective<REDOP>::wait_all_reduce(bool block)
    //--------------------------------------------------------------------------
    {
      return perform_collective_wait(block);
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    typename REDOP::RHS AllReduceCollective<REDOP>::sync_all_reduce(
                                                        typename REDOP::RHS val)
    //--------------------------------------------------------------------------
    {
      async_all_reduce(val);
      return get_result();
    }

    //--------------------------------------------------------------------------
    template<typename REDOP>
    typename REDOP::RHS AllReduceCollective<REDOP>::get_result(void)
    //--------------------------------------------------------------------------
    {
      // Wait for the results to be ready
      wait_all_reduce(true);
      // Need to avoid races here so we have to always recompute the last stage
      typename REDOP::RHS result = value;
      if (!future_values.empty())
      {
#ifdef DEBUG_LEGION
        // Should be at most one stage left
        assert(future_values.size() == 1);
#endif
        const typename std::map<int,std::vector<typename REDOP::RHS> >::
          const_iterator last = future_values.begin();
        if (last->first == -1)
        {
          // Special case for the last stage which already includes our
          // value so just do the overwrite
#ifdef DEBUG_LEGION
          assert(last->second.size() == 1);
#endif
          result = last->second.front();
        }
        else
        {
          // Do the reduction here
          for (typename std::vector<typename REDOP::RHS>::const_iterator it =
                last->second.begin(); it != last->second.end(); it++)
            REDOP::template fold<true/*exclusive*/>(result, *it);
        }
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Barrier Exchange Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename BAR>
    BarrierExchangeCollective<BAR>::BarrierExchangeCollective(
        ReplicateContext *ctx, size_t win_size, 
        typename std::vector<BAR> &bars, CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx), window_size(win_size), barriers(bars)
    //--------------------------------------------------------------------------
    { 
    }

    //--------------------------------------------------------------------------
    template<typename BAR>
    BarrierExchangeCollective<BAR>::BarrierExchangeCollective(
                                           const BarrierExchangeCollective &rhs)
      : AllGatherCollective(rhs), window_size(0), barriers(rhs.barriers)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename BAR>
    BarrierExchangeCollective<BAR>::~BarrierExchangeCollective(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename BAR>
    BarrierExchangeCollective<BAR>& BarrierExchangeCollective<BAR>::operator=(
                                           const BarrierExchangeCollective &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename BAR>
    void BarrierExchangeCollective<BAR>::exchange_barriers_async(void)
    //--------------------------------------------------------------------------
    {
      // First make our local barriers and put them in the data structure
      {
        AutoLock c_lock(collective_lock);
        for (unsigned index = local_shard; 
              index < window_size; index += manager->total_shards)
        {
#ifdef DEBUG_LEGION
          assert(local_barriers.find(index) == local_barriers.end());
#endif
          local_barriers[index] = 
              BAR(Realm::Barrier::create_barrier(manager->total_shards));
        }
      }
      // Now we can start the exchange from this shard 
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    template<typename BAR>
    void BarrierExchangeCollective<BAR>::wait_for_barrier_exchange(void)
    //--------------------------------------------------------------------------
    {
      // Wait for everything to be done
      perform_collective_wait();
#ifdef DEBUG_LEGION
      assert(local_barriers.size() == window_size);
#endif
      // Fill in the barrier vector with the barriers we've got from everyone
      barriers.resize(window_size);
      for (typename std::map<unsigned,BAR>::const_iterator it = 
            local_barriers.begin(); it != local_barriers.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->first < window_size);
#endif
        barriers[it->first] = it->second;
      }
    }

    //--------------------------------------------------------------------------
    template<typename BAR>
    void BarrierExchangeCollective<BAR>::pack_collective_stage(Serializer &rez, 
                                                               int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize(window_size);
      rez.serialize<size_t>(local_barriers.size());
      for (typename std::map<unsigned,BAR>::const_iterator it = 
            local_barriers.begin(); it != local_barriers.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    template<typename BAR>
    void BarrierExchangeCollective<BAR>::unpack_collective_stage(
                                                 Deserializer &derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t other_window_size;
      derez.deserialize(other_window_size);
      if (other_window_size != window_size)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Context configurations for control replicated "
                      "task %s were assigned different maximum window sizes "
                      "of %zd and %zd by the mapper which is illegal.",
                      context->owner_task->get_task_name(), window_size,
                      other_window_size)
      size_t num_bars;
      derez.deserialize(num_bars);
      for (unsigned idx = 0; idx < num_bars; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        derez.deserialize(local_barriers[index]);
      }
    } 

    // Explicit instantiation of our two kinds of barriers
    template class BarrierExchangeCollective<RtBarrier>;
    template class BarrierExchangeCollective<ApBarrier>;

    /////////////////////////////////////////////////////////////
    // Buffer Broadcast
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    void BufferBroadcast::broadcast(void *b, size_t s, bool copy)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(buffer == NULL);
#endif
      if (copy)
      {
        size = s;
        buffer = malloc(size);
        memcpy(buffer, b, size);
        own = true;
      }
      else
      {
        buffer = b;
        size = s;
        own = false;
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    const void* BufferBroadcast::get_buffer(size_t &s, bool wait)
    //--------------------------------------------------------------------------
    {
      if (wait) 
        perform_collective_wait();
      s = size;
      return buffer;
    }

    //--------------------------------------------------------------------------
    void BufferBroadcast::pack_collective(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(size);
      if (size > 0)
        rez.serialize(buffer, size);
    }

    //--------------------------------------------------------------------------
    void BufferBroadcast::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(size);
      if (size > 0)
      {
#ifdef DEBUG_LEGION
        assert(buffer == NULL);
#endif
        buffer = malloc(size);  
        derez.deserialize(buffer, size);
        own = true;
      }
    }

    /////////////////////////////////////////////////////////////
    // Shard Sync Tree 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardSyncTree::ShardSyncTree(ReplicateContext *ctx, ShardID origin,
                                 CollectiveIndexLocation loc)
      : BroadcastCollective(loc, ctx, origin), 
        is_origin(origin == ctx->owner_shard->shard_id)
    //--------------------------------------------------------------------------
    {
      if (is_origin)
      {
        // All we need to do is the broadcast and then wait for 
        // everything to be done
        perform_collective_async();
        // Now wait for the result to be ready
        if (!done_preconditions.empty())
        {
          RtEvent ready = Runtime::merge_events(done_preconditions);
          ready.wait();
        }
      }
    }

    //--------------------------------------------------------------------------
    ShardSyncTree::~ShardSyncTree(void)
    //--------------------------------------------------------------------------
    {
      if (!is_origin)
      {
        // Perform the collective wait
        perform_collective_wait();
        // Trigger our done event when all the preconditions are ready
#ifdef DEBUG_LEGION
        assert(done_event.exists());
#endif
        if (!done_preconditions.empty())
          Runtime::trigger_event(done_event,
              Runtime::merge_events(done_preconditions));
        else
          Runtime::trigger_event(done_event);
      }
    }

    //--------------------------------------------------------------------------
    void ShardSyncTree::pack_collective(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      RtUserEvent next = Runtime::create_rt_user_event();
      rez.serialize(next);
      done_preconditions.insert(next);
    }

    //--------------------------------------------------------------------------
    void ShardSyncTree::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(done_event);
    }

    /////////////////////////////////////////////////////////////
    // Shard Event Tree 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardEventTree::ShardEventTree(ReplicateContext *ctx, ShardID origin,
                                   CollectiveID id)
      : BroadcastCollective(ctx, id, origin),
        is_origin(origin == ctx->owner_shard->shard_id)
    //--------------------------------------------------------------------------
    {
      if (!is_origin)
      {
        local_event = Runtime::create_rt_user_event();
        trigger_event = local_event;
      }
    }

    //--------------------------------------------------------------------------
    ShardEventTree::~ShardEventTree(void)
    //--------------------------------------------------------------------------
    {
      if (finished_event.exists() && !finished_event.has_triggered())
        finished_event.wait();
    }

    //--------------------------------------------------------------------------
    void ShardEventTree::signal_tree(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_origin);
      assert(!trigger_event.exists());
#endif
      trigger_event = precondition;
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    RtEvent ShardEventTree::get_local_event(void)
    //--------------------------------------------------------------------------
    {
      finished_event = perform_collective_wait(false/*block*/); 
      return local_event;
    }

    //--------------------------------------------------------------------------
    void ShardEventTree::pack_collective(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(trigger_event);
    }

    //--------------------------------------------------------------------------
    void ShardEventTree::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_event.exists());
#endif
      RtEvent precondition;
      derez.deserialize(precondition);
      Runtime::trigger_event(local_event, precondition);
    }

    /////////////////////////////////////////////////////////////
    // Cross Product Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CrossProductCollective::CrossProductCollective(ReplicateContext *ctx,
                                                   CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CrossProductCollective::CrossProductCollective(
                                              const CrossProductCollective &rhs)
      : AllGatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    CrossProductCollective::~CrossProductCollective(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    CrossProductCollective& CrossProductCollective::operator=(
                                              const CrossProductCollective &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void CrossProductCollective::exchange_partitions(
                                   std::map<IndexSpace,IndexPartition> &handles)
    //--------------------------------------------------------------------------
    {
      // Need the lock in case we are unpacking other things here
      {
        AutoLock c_lock(collective_lock);
        // Only put the non-empty partitions into our local set
        for (std::map<IndexSpace,IndexPartition>::const_iterator it = 
              handles.begin(); it != handles.end(); it++)
        {
          if (!it->second.exists())
            continue;
          non_empty_handles.insert(*it);
        }
      }
      // Now we do the exchange
      perform_collective_sync();
      // When we wake up we should have all the handles and no need the lock
      // to access them
#ifdef DEBUG_LEGION
      assert(handles.size() == non_empty_handles.size());
#endif
      handles = non_empty_handles;
    }

    //--------------------------------------------------------------------------
    void CrossProductCollective::pack_collective_stage(Serializer &rez, 
                                                       int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(non_empty_handles.size());
      for (std::map<IndexSpace,IndexPartition>::const_iterator it = 
            non_empty_handles.begin(); it != non_empty_handles.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void CrossProductCollective::unpack_collective_stage(Deserializer &derez,
                                                         int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_handles;
      derez.deserialize(num_handles);
      for (unsigned idx = 0; idx < num_handles; idx++)
      {
        IndexSpace handle;
        derez.deserialize(handle);
        derez.deserialize(non_empty_handles[handle]);
      }
    }

    /////////////////////////////////////////////////////////////
    // Sharding Gather Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardingGatherCollective::ShardingGatherCollective(ReplicateContext *ctx,
                                   ShardID target, CollectiveIndexLocation loc)
      : GatherCollective(loc, ctx, target)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    ShardingGatherCollective::ShardingGatherCollective(
                                            const ShardingGatherCollective &rhs)
      : GatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShardingGatherCollective::~ShardingGatherCollective(void)
    //--------------------------------------------------------------------------
    {
      // Make sure that we wait in case we still have messages to pass on
      perform_collective_wait();
    }

    //--------------------------------------------------------------------------
    ShardingGatherCollective& ShardingGatherCollective::operator=(
                                            const ShardingGatherCollective &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ShardingGatherCollective::pack_collective(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(results.size());
      for (std::map<ShardID,ShardingID>::const_iterator it = 
            results.begin(); it != results.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void ShardingGatherCollective::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_results;
      derez.deserialize(num_results);
      for (unsigned idx = 0; idx < num_results; idx++)
      {
        ShardID shard;
        derez.deserialize(shard);
        derez.deserialize(results[shard]);
      }
    }

    //--------------------------------------------------------------------------
    void ShardingGatherCollective::contribute(ShardingID value)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
#ifdef DEBUG_LEGION
        assert(results.find(local_shard) == results.end());
#endif
        results[local_shard] = value;
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    bool ShardingGatherCollective::validate(ShardingID value)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_target());
#endif
      // Wait for the results
      perform_collective_wait();
      for (std::map<ShardID,ShardingID>::const_iterator it = 
            results.begin(); it != results.end(); it++)
      {
        if (it->second != value)
          return false;
      }
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Indirect Record Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndirectRecordExchange::IndirectRecordExchange(ReplicateContext *ctx,
                                                   CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndirectRecordExchange::IndirectRecordExchange(
                                              const IndirectRecordExchange &rhs)
      : AllGatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndirectRecordExchange::~IndirectRecordExchange(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndirectRecordExchange& IndirectRecordExchange::operator=(
                                              const IndirectRecordExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndirectRecordExchange::exchange_records(
                           LegionVector<IndirectRecord>::aligned &local_records)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(records.empty());
#endif
      for (LegionVector<IndirectRecord>::aligned::const_iterator it = 
            local_records.begin(); it != local_records.end(); it++)
      {
        const IndirectKey key(it->inst, it->ready_event, it->domain);
        records[key] = it->fields;
      }
      perform_collective_sync();
      local_records.resize(records.size());
      unsigned index = 0;
      for (LegionMap<IndirectKey,FieldMask>::aligned::const_iterator it = 
            records.begin(); it != records.end(); it++, index++)
      {
        IndirectRecord &record = local_records[index];
        record.inst = it->first.inst;
        record.ready_event = it->first.ready_event;
        record.domain = it->first.domain;
        record.fields = it->second;
      }
    }

    //--------------------------------------------------------------------------
    void IndirectRecordExchange::pack_collective_stage(Serializer &rez,
                                                       int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize(records.size());
      for (LegionMap<IndirectKey,FieldMask>::aligned::const_iterator it = 
            records.begin(); it != records.end(); it++)
      {
        rez.serialize(it->first.inst);
        rez.serialize(it->first.ready_event);
        rez.serialize(it->first.domain);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void IndirectRecordExchange::unpack_collective_stage(Deserializer &derez,
                                                         int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_records;
      derez.deserialize(num_records);
      for (unsigned idx = 0; idx < num_records; idx++)
      {
        IndirectKey key;
        derez.deserialize(key.inst);
        derez.deserialize(key.ready_event);
        derez.deserialize(key.domain);
        LegionMap<IndirectKey,FieldMask>::aligned::iterator finder = 
          records.find(key);
        if (finder != records.end())
        {
          FieldMask mask;
          derez.deserialize(mask);
          finder->second |= mask;
        }
        else
          derez.deserialize(records[key]);
      }
    }

    /////////////////////////////////////////////////////////////
    // Field Descriptor Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldDescriptorExchange::FieldDescriptorExchange(ReplicateContext *ctx,
                                                   CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldDescriptorExchange::FieldDescriptorExchange(
                                             const FieldDescriptorExchange &rhs)
      : AllGatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FieldDescriptorExchange::~FieldDescriptorExchange(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldDescriptorExchange& FieldDescriptorExchange::operator=(
                                             const FieldDescriptorExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    ApEvent FieldDescriptorExchange::exchange_descriptors(ApEvent ready_event,
                                  const std::vector<FieldDataDescriptor> &descs)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
        ready_events.insert(ready_event);
        descriptors.insert(descriptors.end(), descs.begin(), descs.end());
        if (participating)
        {
          remote_to_trigger.resize(shard_collective_stages + 1);
          local_preconditions.resize(shard_collective_stages + 1);
        }
        else
        {
          remote_to_trigger.resize(1);
          local_preconditions.resize(1);
        }
      }
      perform_collective_sync();
      return Runtime::merge_events(NULL, ready_events);
    }

    //--------------------------------------------------------------------------
    ApEvent FieldDescriptorExchange::exchange_completion(ApEvent complete)
    //--------------------------------------------------------------------------
    {
      if (participating)
      {
        // Might have a precondition from a remainder shard 
        if (!local_preconditions[0].empty())
        {
#ifdef DEBUG_LEGION
          assert(local_preconditions[0].size() == 1);
#endif
          complete = Runtime::merge_events(NULL, complete,
              *(local_preconditions[0].begin()));
        }
        const std::set<ApUserEvent> &to_trigger = remote_to_trigger[0];
        for (std::set<ApUserEvent>::const_iterator it = 
              to_trigger.begin(); it != to_trigger.end(); it++)
          Runtime::trigger_event(NULL, *it, complete);
        const ApEvent done = 
          Runtime::merge_events(NULL, local_preconditions.back());
        // If we have a remainder shard then we need to signal them too
        if (!remote_to_trigger[shard_collective_stages].empty())
        {
#ifdef DEBUG_LEGION
          assert(remote_to_trigger[shard_collective_stages].size() == 1);
#endif
          Runtime::trigger_event(NULL,
              *(remote_to_trigger[shard_collective_stages].begin()), done);     
        }
        return done;
      }
      else
      {
        // Not participating so we should have exactly one thing to 
        // trigger and one precondition for being done
#ifdef DEBUG_LEGION
        assert(remote_to_trigger[0].size() == 1);
        assert(local_preconditions[0].size() == 1);
#endif
        Runtime::trigger_event(NULL, *(remote_to_trigger[0].begin()), complete);
        return *(local_preconditions[0].begin());
      }
    }

    //--------------------------------------------------------------------------
    void FieldDescriptorExchange::pack_collective_stage(Serializer &rez,
                                                        int stage)
    //--------------------------------------------------------------------------
    {
      // Always make a stage precondition and send it back
      ApUserEvent stage_complete = Runtime::create_ap_user_event(NULL);
      rez.serialize(stage_complete);
      if (stage == -1)
      {
#ifdef DEBUG_LEGION
        assert(!local_preconditions.empty());
        assert(local_preconditions[0].empty());
#endif
        // Always save this as a precondition for later
        local_preconditions[0].insert(stage_complete);
      }
      else 
      {
#ifdef DEBUG_LEGION
        assert(participating);
        assert(stage < shard_collective_stages);
#endif
        std::set<ApEvent> &preconditions = 
          local_preconditions[shard_collective_stages - stage];
        preconditions.insert(stage_complete);
        // See if we've sent all our messages in which case we can 
        // trigger all the remote user events for any previous stages
        if (((stage == (shard_collective_stages-1)) && 
              (int(preconditions.size()) == shard_collective_last_radix)) ||
            ((stage < (shard_collective_stages-1)) &&
              (int(preconditions.size()) == shard_collective_radix)))
        {
          const std::set<ApUserEvent> &to_trigger = 
           remote_to_trigger[(stage > 0) ? (stage-1) : shard_collective_stages];
          // Check for empty which can happen with stage 0 if there
          // are no remainders
          if (!to_trigger.empty())
          {
            const ApEvent stage_pre = Runtime::merge_events(NULL,preconditions);
            for (std::set<ApUserEvent>::const_iterator it = 
                  to_trigger.begin(); it != to_trigger.end(); it++)
              Runtime::trigger_event(NULL, *it, stage_pre);
          }
        }
      }
      rez.serialize<size_t>(ready_events.size());
      for (std::set<ApEvent>::const_iterator it = ready_events.begin();
            it != ready_events.end(); it++)
        rez.serialize(*it);
      rez.serialize<size_t>(descriptors.size());
      for (std::vector<FieldDataDescriptor>::const_iterator it = 
            descriptors.begin(); it != descriptors.end(); it++)
        rez.serialize(*it);
    }

    //--------------------------------------------------------------------------
    void FieldDescriptorExchange::unpack_collective_stage(Deserializer &derez,
                                                          int stage)
    //--------------------------------------------------------------------------
    {
      ApUserEvent remote_complete;
      derez.deserialize(remote_complete);
      if (stage == -1)
      {
#ifdef DEBUG_LEGION
        assert(!remote_to_trigger.empty());
#endif
        if (participating)
        {
#ifdef DEBUG_LEGION
          assert(remote_to_trigger[shard_collective_stages].empty());
#endif
          remote_to_trigger[shard_collective_stages].insert(remote_complete);
        }
        else
        {
#ifdef DEBUG_LEGION
          assert(remote_to_trigger[0].empty());
#endif
          remote_to_trigger[0].insert(remote_complete);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(participating);
        assert(stage < int(remote_to_trigger.size()));
#endif
        remote_to_trigger[stage].insert(remote_complete);
      }
      size_t num_events;
      derez.deserialize(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        ApEvent ready;
        derez.deserialize(ready);
        ready_events.insert(ready);
      }
      unsigned offset = descriptors.size();
      size_t num_descriptors;
      derez.deserialize(num_descriptors);
      descriptors.resize(offset + num_descriptors);
      for (unsigned idx = 0; idx < num_descriptors; idx++)
        derez.deserialize(descriptors[offset + idx]);
    }

    /////////////////////////////////////////////////////////////
    // Field Descriptor Gather 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldDescriptorGather::FieldDescriptorGather(ReplicateContext *ctx,
                             ShardID target, CollectiveIndexLocation loc)
      : GatherCollective(loc, ctx, target), used(false)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FieldDescriptorGather::FieldDescriptorGather(
                                               const FieldDescriptorGather &rhs)
      : GatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FieldDescriptorGather::~FieldDescriptorGather(void)
    //--------------------------------------------------------------------------
    {
      // Make sure that we wait in case we still have messages to pass on
      if (used)
        perform_collective_wait();
#ifdef DEBUG_LEGION
      assert(!complete_event.exists() || complete_event.has_triggered());
#endif
    }

    //--------------------------------------------------------------------------
    FieldDescriptorGather& FieldDescriptorGather::operator=(
                                               const FieldDescriptorGather &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FieldDescriptorGather::pack_collective(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(complete_event.exists());
#endif
      // Trigger any remote complete events we have dependent on our event
      if (!remote_complete_events.empty())
      {
        for (std::set<ApUserEvent>::const_iterator it = 
              remote_complete_events.begin(); it != 
              remote_complete_events.end(); it++)
          Runtime::trigger_event(NULL, *it, complete_event); 
      }
      rez.serialize(complete_event);
      rez.serialize<size_t>(ready_events.size());
      for (std::set<ApEvent>::const_iterator it = ready_events.begin();
            it != ready_events.end(); it++)
        rez.serialize(*it);
      rez.serialize<size_t>(descriptors.size());
      for (std::vector<FieldDataDescriptor>::const_iterator it = 
            descriptors.begin(); it != descriptors.end(); it++)
        rez.serialize(*it);
    }
    
    //--------------------------------------------------------------------------
    void FieldDescriptorGather::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      ApUserEvent remote_complete;
      derez.deserialize(remote_complete);
      remote_complete_events.insert(remote_complete);
      size_t num_events;
      derez.deserialize(num_events);
      for (unsigned idx = 0; idx < num_events; idx++)
      {
        ApEvent ready;
        derez.deserialize(ready);
        ready_events.insert(ready);
      }
      unsigned offset = descriptors.size();
      size_t num_descriptors;
      derez.deserialize(num_descriptors);
      descriptors.resize(offset + num_descriptors);
      for (unsigned idx = 0; idx < num_descriptors; idx++)
        derez.deserialize(descriptors[offset + idx]);
    }

    //--------------------------------------------------------------------------
    void FieldDescriptorGather::contribute(ApEvent ready_event,
                                  const std::vector<FieldDataDescriptor> &descs)
    //--------------------------------------------------------------------------
    {
      used = true;
      {
        AutoLock c_lock(collective_lock);
        ready_events.insert(ready_event);
        descriptors.insert(descriptors.end(), descs.begin(), descs.end());
        // If we're not the owner make our complete event
#ifdef DEBUG_LEGION
        assert(!complete_event.exists());
#endif
        if (!is_target())
          complete_event = Runtime::create_ap_user_event(NULL);
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    const std::vector<FieldDataDescriptor>& 
                     FieldDescriptorGather::get_full_descriptors(ApEvent &ready)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      ready = Runtime::merge_events(NULL, ready_events);
      return descriptors;
    }

    //--------------------------------------------------------------------------
    void FieldDescriptorGather::notify_remote_complete(ApEvent precondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_target());
#endif
      if (!remote_complete_events.empty())
      {
        for (std::set<ApUserEvent>::const_iterator it = 
              remote_complete_events.begin(); it != 
              remote_complete_events.end(); it++)
          Runtime::trigger_event(NULL, *it, precondition);
      }
    }

    //--------------------------------------------------------------------------
    ApEvent FieldDescriptorGather::get_complete_event(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_target());
      assert(complete_event.exists());
#endif
      return complete_event;
    }

    /////////////////////////////////////////////////////////////
    // Future Broadcast 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureBroadcast::FutureBroadcast(ReplicateContext *ctx, CollectiveID id,
                                     ShardID source, FutureImpl *i)
      : BroadcastCollective(ctx, id, source), impl(i)
    //--------------------------------------------------------------------------
    {
      if (source == ctx->owner_shard->shard_id)
        ready = impl->subscribe_internal();
    }

    //--------------------------------------------------------------------------
    FutureBroadcast::FutureBroadcast(const FutureBroadcast &rhs)
      : BroadcastCollective(rhs), impl(rhs.impl)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FutureBroadcast::~FutureBroadcast(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureBroadcast& FutureBroadcast::operator=(const FutureBroadcast &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FutureBroadcast::pack_collective(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      const size_t result_size = impl->get_untyped_size(true/*internal*/); 
      rez.serialize(result_size);
      if (result_size > 0)
        rez.serialize(impl->get_untyped_result(true, NULL, true), result_size);
    }

    //--------------------------------------------------------------------------
    void FutureBroadcast::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t result_size;
      derez.deserialize(result_size);
      if (result_size > 0)
      {
        const void *ptr = derez.get_current_pointer();  
        impl->set_result(ptr, result_size, false/*owned*/);
        derez.advance_pointer(result_size);
      }
      else
        impl->set_result(NULL, 0, false/*owned*/);
    }

    //--------------------------------------------------------------------------
    void FutureBroadcast::broadcast_future(void)
    //--------------------------------------------------------------------------
    {
      if (ready.exists() && !ready.has_triggered())
        ready.wait();
      perform_collective_async();
    }

    /////////////////////////////////////////////////////////////
    // Future Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureExchange::FutureExchange(ReplicateContext *ctx, size_t size,
                                   CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx), future_size(size)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureExchange::FutureExchange(const FutureExchange &rhs)
      : AllGatherCollective(rhs), future_size(0)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FutureExchange::~FutureExchange(void)
    //--------------------------------------------------------------------------
    {
      // Delete all the futures except our local shard one since we know
      // that we don't actually own that memory
      for (std::map<ShardID,void*>::const_iterator it = results.begin();
            it != results.end(); it++)
        free(it->second);
    }

    //--------------------------------------------------------------------------
    FutureExchange& FutureExchange::operator=(const FutureExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FutureExchange::pack_collective_stage(Serializer &rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(results.size());
      for (std::map<ShardID,void*>::const_iterator it = results.begin();
            it != results.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second, future_size);
      }
    }

    //--------------------------------------------------------------------------
    void FutureExchange::unpack_collective_stage(Deserializer &derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_results;
      derez.deserialize(num_results);
      for (unsigned idx = 0; idx < num_results; idx++)
      {
        ShardID shard;
        derez.deserialize(shard);
        if (results.find(shard) != results.end())
        {
          derez.advance_pointer(future_size);
          continue;
        }
        void *buffer = malloc(future_size);
        derez.deserialize(buffer, future_size);
        results[shard] = buffer;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent FutureExchange::exchange_futures(void *value)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
#ifdef DEBUG_LEGION
        assert(results.find(local_shard) == results.end());
#endif
        results[local_shard] = value;
      }
      perform_collective_async();
      return perform_collective_wait(false/*block*/);
    }

    //--------------------------------------------------------------------------
    void FutureExchange::reduce_futures(ReplIndexTask *target)
    //--------------------------------------------------------------------------
    {
      // Now we apply the shard results in order to ensure that we get
      // the same bitwise order across all the shards
      // No need for the lock anymore since we know we're done
      for (std::map<ShardID,void*>::const_iterator it = results.begin();
            it != results.end(); it++)
        target->fold_reduction_future(it->second, future_size, 
                                      false/*owner*/, true/*exclusive*/);
    }

    //--------------------------------------------------------------------------
    void FutureExchange::reduce_futures(const ReductionOp *redop,
                                        void *result_buffer)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(future_size == redop->sizeof_rhs);
#endif
      redop->init(result_buffer, 1/*count*/);
      // Now we apply the shard results in order to ensure that we get
      // the same bitwise order across all the shards
      // No need for the lock anymore since we know we're done
      for (std::map<ShardID,void*>::const_iterator it = results.begin();
            it != results.end(); it++)
        redop->fold(result_buffer, it->second, 1/*count*/, true/*exclusive*/);
    }

    /////////////////////////////////////////////////////////////
    // Future Name Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureNameExchange::FutureNameExchange(ReplicateContext *ctx,
                   CollectiveID id, ReplFutureMapImpl *m, ReferenceMutator *mut)
      : AllGatherCollective(ctx, id), future_map(m), mutator(mut)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureNameExchange::FutureNameExchange(const FutureNameExchange &rhs)
      : AllGatherCollective(rhs), future_map(rhs.future_map), 
        mutator(rhs.mutator)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FutureNameExchange::~FutureNameExchange(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureNameExchange& FutureNameExchange::operator=(
                                                  const FutureNameExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void FutureNameExchange::pack_collective_stage(Serializer &rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(results.size());
      for (std::map<DomainPoint,Future>::const_iterator it = 
            results.begin(); it != results.end(); it++)
      {
        rez.serialize(it->first);
        if (it->second.impl != NULL)
          it->second.impl->pack_future(rez);
        else
          rez.serialize<DistributedID>(0);
      }
    }

    //--------------------------------------------------------------------------
    void FutureNameExchange::unpack_collective_stage(Deserializer &derez,
                                                     int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_futures;
      derez.deserialize(num_futures);
      Runtime *runtime = context->runtime;
      for (unsigned idx = 0; idx < num_futures; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        FutureImpl *impl = FutureImpl::unpack_future(runtime, derez, mutator);
        if (impl != NULL)
        {
          // Add the reference ourselves so we can capture the effects
          impl->add_base_gc_ref(FUTURE_HANDLE_REF, mutator);
          results[point] = Future(impl, false/*need referece*/);
        }
        else
          results[point] = Future();
      }
    }

    //--------------------------------------------------------------------------
    void FutureNameExchange::exchange_future_names(
                                          std::map<DomainPoint,Future> &futures)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
        results.insert(futures.begin(), futures.end());
      }
      perform_collective_sync();
      futures = results;
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Processor Broadcast 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochMappingBroadcast::MustEpochMappingBroadcast(
            ReplicateContext *ctx, ShardID origin, CollectiveID collective_id)
      : BroadcastCollective(ctx, collective_id, origin)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochMappingBroadcast::MustEpochMappingBroadcast(
                                           const MustEpochMappingBroadcast &rhs)
      : BroadcastCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MustEpochMappingBroadcast::~MustEpochMappingBroadcast(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_done_event.exists());
#endif
      if (!done_events.empty())
        Runtime::trigger_event(local_done_event,
            Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(local_done_event);
      // This should only happen on the owner node
      if (!held_references.empty())
      {
        // Wait for all the other shards to be done
        local_done_event.wait();
        // Now we can remove our held references
        for (std::set<PhysicalManager*>::const_iterator it = 
              held_references.begin(); it != held_references.end(); it++)
          if ((*it)->remove_base_valid_ref(REPLICATION_REF))
            delete (*it);
      }
    }
    
    //--------------------------------------------------------------------------
    MustEpochMappingBroadcast& MustEpochMappingBroadcast::operator=(
                                           const MustEpochMappingBroadcast &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingBroadcast::pack_collective(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      RtUserEvent next_done = Runtime::create_rt_user_event();
      done_events.insert(next_done);
      rez.serialize(next_done);
      rez.serialize<size_t>(processors.size());
      for (unsigned idx = 0; idx < processors.size(); idx++)
        rez.serialize(processors[idx]);
      rez.serialize<size_t>(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const std::vector<DistributedID> &dids = instances[idx];
        rez.serialize<size_t>(dids.size());
        for (std::vector<DistributedID>::const_iterator it = 
              dids.begin(); it != dids.end(); it++)
          rez.serialize(*it);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingBroadcast::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(local_done_event);
      size_t num_procs;
      derez.deserialize(num_procs);
      processors.resize(num_procs);
      for (unsigned idx = 0; idx < num_procs; idx++)
        derez.deserialize(processors[idx]);
      size_t num_constraints;
      derez.deserialize(num_constraints);
      instances.resize(num_constraints);
      for (unsigned idx1 = 0; idx1 < num_constraints; idx1++)
      {
        size_t num_dids;
        derez.deserialize(num_dids);
        std::vector<DistributedID> &dids = instances[idx1];
        dids.resize(num_dids);
        for (unsigned idx2 = 0; idx2 < num_dids; idx2++)
          derez.deserialize(dids[idx2]);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingBroadcast::broadcast(
           const std::vector<Processor> &processor_mapping,
           const std::vector<std::vector<Mapping::PhysicalInstance> > &mappings)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!local_done_event.exists());
#endif
      local_done_event = Runtime::create_rt_user_event();
      processors = processor_mapping;
      instances.resize(mappings.size());
      // Add valid references to all the physical instances that we will
      // hold until all the must epoch operations are done with the exchange
      WrapperReferenceMutator mutator(done_events);
      for (unsigned idx1 = 0; idx1 < mappings.size(); idx1++)
      {
        std::vector<DistributedID> &dids = instances[idx1];
        dids.resize(mappings[idx1].size());
        for (unsigned idx2 = 0; idx2 < dids.size(); idx2++)
        {
          const Mapping::PhysicalInstance &inst = mappings[idx1][idx2];
          PhysicalManager *manager = inst.impl->as_instance_manager();
          dids[idx2] = manager->did;
          if (held_references.find(manager) != held_references.end())
            continue;
          manager->add_base_valid_ref(REPLICATION_REF, &mutator);
          held_references.insert(manager);
        }
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingBroadcast::receive_results(
                std::vector<Processor> &processor_mapping,
                const std::vector<unsigned> &constraint_indexes,
                std::vector<std::vector<Mapping::PhysicalInstance> > &mappings,
                std::map<PhysicalManager*,unsigned> &acquired)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      // Just grab all the processors since we still need them
      processor_mapping = processors;
      // We are a little smarter with the mappings since we know exactly
      // which ones we are actually going to need for our local points
      std::set<RtEvent> ready_events;
      Runtime *runtime = manager->runtime;
      for (std::vector<unsigned>::const_iterator it = 
            constraint_indexes.begin(); it != constraint_indexes.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert((*it) < instances.size());
        assert((*it) < mappings.size());
#endif
        const std::vector<DistributedID> &dids = instances[*it];
        std::vector<Mapping::PhysicalInstance> &mapping = mappings[*it];
        mapping.resize(dids.size());
        for (unsigned idx = 0; idx < dids.size(); idx++)
        {
          RtEvent ready;
          mapping[idx].impl = 
            runtime->find_or_request_instance_manager(dids[idx], ready);
          if (!ready.has_triggered())
            ready_events.insert(ready);   
        }
      }
      // Have to wait for the ready events to trigger before we can add
      // our references safely
      if (!ready_events.empty())
      {
        RtEvent ready = Runtime::merge_events(ready_events);
        if (!ready.has_triggered())
          ready.wait();
      }
      // Lastly we need to put acquire references on any of local instances
      WrapperReferenceMutator mutator(done_events);
      for (unsigned idx = 0; idx < constraint_indexes.size(); idx++)
      {
        const unsigned constraint_index = constraint_indexes[idx];
        const std::vector<Mapping::PhysicalInstance> &mapping = 
          mappings[constraint_index];
        // Also grab an acquired reference to these instances
        for (std::vector<Mapping::PhysicalInstance>::const_iterator it = 
              mapping.begin(); it != mapping.end(); it++)
        {
          PhysicalManager *manager = it->impl->as_instance_manager();
          // If we already had a reference to this instance
          // then we don't need to add any additional ones
          if (acquired.find(manager) != acquired.end())
            continue;
          manager->add_base_resource_ref(INSTANCE_MAPPER_REF);
          manager->add_base_valid_ref(MAPPING_ACQUIRE_REF, &mutator);
          acquired[manager] = 1/*count*/; 
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Mapping Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochMappingExchange::MustEpochMappingExchange(ReplicateContext *ctx,
                                                 CollectiveID collective_id)
      : AllGatherCollective(ctx, collective_id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochMappingExchange::MustEpochMappingExchange(
                                            const MustEpochMappingExchange &rhs)
      : AllGatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MustEpochMappingExchange::~MustEpochMappingExchange(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_done_event.exists()); // better have one of these
#endif
      Runtime::trigger_event(local_done_event);
      // See if we need to wait for others to be done before we can
      // remove our valid references
      if (!done_events.empty())
      {
        RtEvent done = Runtime::merge_events(done_events);
        if (!done.has_triggered())
          done.wait();
      }
      // Now we can remove our held references
      for (std::set<PhysicalManager*>::const_iterator it = 
            held_references.begin(); it != held_references.end(); it++)
        if ((*it)->remove_base_valid_ref(REPLICATION_REF))
          delete (*it);
    }
    
    //--------------------------------------------------------------------------
    MustEpochMappingExchange& MustEpochMappingExchange::operator=(
                                            const MustEpochMappingExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingExchange::pack_collective_stage(Serializer &rez,
                                                         int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(processors.size());
      for (std::map<DomainPoint,Processor>::const_iterator it = 
            processors.begin(); it != processors.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
      rez.serialize<size_t>(constraints.size());
      for (std::map<unsigned,ConstraintInfo>::const_iterator it = 
            constraints.begin(); it != constraints.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize<size_t>(it->second.instances.size());
        for (unsigned idx = 0; idx < it->second.instances.size(); idx++)
          rez.serialize(it->second.instances[idx]);
        rez.serialize(it->second.origin_shard);
        rez.serialize(it->second.weight);
      }
      rez.serialize<size_t>(done_events.size());
      for (std::set<RtEvent>::const_iterator it = 
            done_events.begin(); it != done_events.end(); it++)
        rez.serialize(*it);
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingExchange::unpack_collective_stage(Deserializer &derez,
                                                           int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_procs;
      derez.deserialize(num_procs);
      for (unsigned idx = 0; idx < num_procs; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        derez.deserialize(processors[point]);
      }
      size_t num_mappings;
      derez.deserialize(num_mappings);
      for (unsigned idx1 = 0; idx1 < num_mappings; idx1++)
      {
        unsigned constraint_index;
        derez.deserialize(constraint_index);
        std::map<unsigned,ConstraintInfo>::iterator
          finder = constraints.find(constraint_index);
        if (finder == constraints.end())
        {
          // Can unpack directly since we're first
          ConstraintInfo &info = constraints[constraint_index];
          size_t num_dids;
          derez.deserialize(num_dids);
          info.instances.resize(num_dids);
          for (unsigned idx2 = 0; idx2 < num_dids; idx2++)
            derez.deserialize(info.instances[idx2]);
          derez.deserialize(info.origin_shard);
          derez.deserialize(info.weight);
        }
        else
        {
          // Unpack into a temporary
          ConstraintInfo info;
          size_t num_dids;
          derez.deserialize(num_dids);
          info.instances.resize(num_dids);
          for (unsigned idx2 = 0; idx2 < num_dids; idx2++)
            derez.deserialize(info.instances[idx2]);
          derez.deserialize(info.origin_shard);
          derez.deserialize(info.weight);
          // Only keep the result if we have a larger weight
          // or we have the same weight and a smaller shard
          if ((info.weight > finder->second.weight) ||
              ((info.weight == finder->second.weight) &&
               (info.origin_shard < finder->second.origin_shard)))
            finder->second = info;
        }
      }
      size_t num_done;
      derez.deserialize(num_done);
      for (unsigned idx = 0; idx < num_done; idx++)
      {
        RtEvent done_event;
        derez.deserialize(done_event);
        done_events.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingExchange::exchange_must_epoch_mappings(
                ShardID shard_id, size_t total_shards, size_t total_constraints,
                const std::vector<const Task*> &local_tasks,
                const std::vector<const Task*> &all_tasks,
                      std::vector<Processor> &processor_mapping,
                const std::vector<unsigned> &constraint_indexes,
                std::vector<std::vector<Mapping::PhysicalInstance> > &mappings,
                const std::vector<int> &mapping_weights,
                std::map<PhysicalManager*,unsigned> &acquired)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_tasks.size() == processor_mapping.size());
      assert(constraint_indexes.size() == mappings.size());
#endif
      // Add valid references to all the physical instances that we will
      // hold until all the must epoch operations are done with the exchange
      WrapperReferenceMutator mutator(done_events);
      for (unsigned idx = 0; idx < mappings.size(); idx++)
      {
        for (std::vector<Mapping::PhysicalInstance>::const_iterator it = 
              mappings[idx].begin(); it != mappings[idx].end(); it++)
        {
          PhysicalManager *manager = it->impl->as_instance_manager();
          if (held_references.find(manager) != held_references.end())
            continue;
          manager->add_base_valid_ref(REPLICATION_REF, &mutator);
          held_references.insert(manager);
        }
      }
#ifdef DEBUG_LEGION
      assert(!local_done_event.exists());
#endif
      local_done_event = Runtime::create_rt_user_event();
      // Then we can add our instances to the set and do the exchange
      {
        AutoLock c_lock(collective_lock);
        for (unsigned idx = 0; idx < local_tasks.size(); idx++)
        {
          const Task *task = local_tasks[idx];
#ifdef DEBUG_LEGION
          assert(processors.find(task->index_point) == processors.end());
#endif
          processors[task->index_point] = processor_mapping[idx];
        }
        for (unsigned idx1 = 0; idx1 < mappings.size(); idx1++)
        {
          const unsigned constraint_index = constraint_indexes[idx1]; 
#ifdef DEBUG_LEGION
          assert(constraint_index < total_constraints);
#endif
          std::map<unsigned,ConstraintInfo>::iterator
            finder = constraints.find(constraint_index);
          // Only add it if it doesn't exist or it has a lower weight
          // or it has the same weight and is a lower shard
          if ((finder == constraints.end()) || 
              (mapping_weights[idx1] > finder->second.weight) ||
              ((mapping_weights[idx1] == finder->second.weight) &&
               (shard_id < finder->second.origin_shard)))
          {
            ConstraintInfo &info = constraints[constraint_index];
            info.instances.resize(mappings[idx1].size());
            for (unsigned idx2 = 0; idx2 < mappings[idx1].size(); idx2++)
              info.instances[idx2] = mappings[idx1][idx2].impl->did;
            info.origin_shard = shard_id;
            info.weight = mapping_weights[idx1];
          }
        }
        // Also update the local done events
        done_events.insert(local_done_event);
      }
      perform_collective_sync();
      // Start fetching the all the mapping results to get them in flight
      mappings.clear();
      mappings.resize(total_constraints);
      std::set<RtEvent> ready_events;
      Runtime *runtime = manager->runtime;
      // We only need to get the results for local constraints as we 
      // know that we aren't going to care about any of the rest
      for (unsigned idx1 = 0; idx1 < constraint_indexes.size(); idx1++)
      {
        const unsigned constraint_index = constraint_indexes[idx1];
        const std::vector<DistributedID> &dids = 
          constraints[constraint_index].instances;
        std::vector<Mapping::PhysicalInstance> &mapping = 
          mappings[constraint_index];
        mapping.resize(dids.size());
        for (unsigned idx2 = 0; idx2 < dids.size(); idx2++)
        {
          RtEvent ready;
          mapping[idx2].impl = 
            runtime->find_or_request_instance_manager(dids[idx2], ready);
          if (!ready.has_triggered())
            ready_events.insert(ready);   
        }
      }
      // Update the processor mapping
      processor_mapping.resize(all_tasks.size());
      for (unsigned idx = 0; idx < all_tasks.size(); idx++)
      {
        const Task *task = all_tasks[idx];
        std::map<DomainPoint,Processor>::const_iterator finder = 
          processors.find(task->index_point);
#ifdef DEBUG_LEGION
        assert(finder != processors.end());
#endif
        processor_mapping[idx] = finder->second;
      }
      // Wait for all the instances to be ready
      if (!ready_events.empty())
      {
        RtEvent ready = Runtime::merge_events(ready_events);
        if (!ready.has_triggered())
          ready.wait();
      }
      // Lastly we need to put acquire references on any of local instances
      for (unsigned idx = 0; idx < constraint_indexes.size(); idx++)
      {
        const unsigned constraint_index = constraint_indexes[idx];
        const std::vector<Mapping::PhysicalInstance> &mapping = 
          mappings[constraint_index];
        // Also grab an acquired reference to these instances
        for (std::vector<Mapping::PhysicalInstance>::const_iterator it = 
              mapping.begin(); it != mapping.end(); it++)
        {
          PhysicalManager *manager = it->impl->as_instance_manager();
          // If we already had a reference to this instance
          // then we don't need to add any additional ones
          if (acquired.find(manager) != acquired.end())
            continue;
          manager->add_base_resource_ref(INSTANCE_MAPPER_REF);
          manager->add_base_valid_ref(MAPPING_ACQUIRE_REF, &mutator);
          acquired[manager] = 1/*count*/;
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Dependence Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochDependenceExchange::MustEpochDependenceExchange(
                             ReplicateContext *ctx, CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochDependenceExchange::MustEpochDependenceExchange(
                                         const MustEpochDependenceExchange &rhs)
      : AllGatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MustEpochDependenceExchange::~MustEpochDependenceExchange(void)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    MustEpochDependenceExchange& MustEpochDependenceExchange::operator=(
                                         const MustEpochDependenceExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MustEpochDependenceExchange::pack_collective_stage(Serializer &rez,
                                                            int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(mapping_dependences.size());
      for (std::map<DomainPoint,RtUserEvent>::const_iterator it = 
            mapping_dependences.begin(); it != mapping_dependences.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochDependenceExchange::unpack_collective_stage(
                                                 Deserializer &derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_deps;
      derez.deserialize(num_deps);
      for (unsigned idx = 0; idx < num_deps; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        derez.deserialize(mapping_dependences[point]);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochDependenceExchange::exchange_must_epoch_dependences(
                               std::map<DomainPoint,RtUserEvent> &mapped_events)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
        for (std::map<DomainPoint,RtUserEvent>::const_iterator it = 
              mapped_events.begin(); it != mapped_events.end(); it++)
          mapping_dependences.insert(*it);
      }
      perform_collective_sync();
      // No need to hold the lock after the collective is complete
      mapped_events.swap(mapping_dependences);
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Completion Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochCompletionExchange::MustEpochCompletionExchange(
                             ReplicateContext *ctx, CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochCompletionExchange::MustEpochCompletionExchange(
                                         const MustEpochCompletionExchange &rhs)
      : AllGatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MustEpochCompletionExchange::~MustEpochCompletionExchange(void)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    MustEpochCompletionExchange& MustEpochCompletionExchange::operator=(
                                         const MustEpochCompletionExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MustEpochCompletionExchange::pack_collective_stage(Serializer &rez,
                                                            int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(tasks_mapped.size());
      for (std::set<RtEvent>::const_iterator it = 
            tasks_mapped.begin(); it != tasks_mapped.end(); it++)
        rez.serialize(*it);
      rez.serialize<size_t>(tasks_complete.size());
      for (std::set<ApEvent>::const_iterator it = 
            tasks_complete.begin(); it != tasks_complete.end(); it++)
        rez.serialize(*it);
    }

    //--------------------------------------------------------------------------
    void MustEpochCompletionExchange::unpack_collective_stage(
                                                 Deserializer &derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_mapped;
      derez.deserialize(num_mapped);
      for (unsigned idx = 0; idx < num_mapped; idx++)
      {
        RtEvent mapped;
        derez.deserialize(mapped);
        tasks_mapped.insert(mapped);
      }
      size_t num_complete;
      derez.deserialize(num_complete);
      for (unsigned idx = 0; idx < num_complete; idx++)
      {
        ApEvent complete;
        derez.deserialize(complete);
        tasks_complete.insert(complete);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochCompletionExchange::exchange_must_epoch_completion(
                                RtEvent mapped, ApEvent complete,
                                std::set<RtEvent> &all_mapped,
                                std::set<ApEvent> &all_complete)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
        tasks_mapped.insert(mapped);
        tasks_complete.insert(complete);
      }
      perform_collective_sync();
      // No need to hold the lock after the collective is complete
      all_mapped.swap(tasks_mapped);
      all_complete.swap(tasks_complete);
    }

    /////////////////////////////////////////////////////////////
    // Sharded Mapping Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardedMappingExchange::ShardedMappingExchange(CollectiveIndexLocation loc,
                            ReplicateContext *ctx, ShardID sid, bool check_map)
      : AllGatherCollective(loc, ctx), shard_id(sid), check_mappings(check_map)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShardedMappingExchange::ShardedMappingExchange(
                                                const ShardedMappingExchange &i)
      : AllGatherCollective(i), shard_id(0), check_mappings(false)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ShardedMappingExchange::~ShardedMappingExchange(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ShardedMappingExchange& ShardedMappingExchange::operator=(
                                              const ShardedMappingExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ShardedMappingExchange::pack_collective_stage(Serializer &rez, 
                                                       int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(mappings.size());
      for (std::map<DistributedID,LegionMap<ShardID,FieldMask>::aligned>::
            const_iterator mit = mappings.begin(); 
            mit != mappings.end(); mit++)
      {
        rez.serialize(mit->first);
        rez.serialize<size_t>(mit->second.size());
        for (LegionMap<ShardID,FieldMask>::aligned::const_iterator it = 
              mit->second.begin(); it != mit->second.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      rez.serialize<size_t>(global_views.size());
      for (LegionMap<DistributedID,FieldMask>::aligned::const_iterator it = 
            global_views.begin(); it != global_views.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void ShardedMappingExchange::unpack_collective_stage(Deserializer &derez,
                                                         int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_mappings;
      derez.deserialize(num_mappings);
      for (unsigned idx1 = 0; idx1 < num_mappings; idx1++)
      {
        DistributedID did;
        derez.deserialize(did);
        size_t num_shards;
        derez.deserialize(num_shards);
        LegionMap<ShardID,FieldMask>::aligned &inst_map = mappings[did];
        for (unsigned idx2 = 0; idx2 < num_shards; idx2++)
        {
          ShardID sid;
          derez.deserialize(sid);
          LegionMap<ShardID,FieldMask>::aligned::iterator finder = 
            inst_map.find(sid);
          if (finder != inst_map.end())
          {
            FieldMask mask;
            derez.deserialize(mask);
            finder->second |= mask;
          }
          else
            derez.deserialize(inst_map[sid]);
        }
      }
      size_t num_views;
      derez.deserialize(num_views);
      for (unsigned idx = 0; idx < num_views; idx++)
      {
        DistributedID did;
        derez.deserialize(did);
        LegionMap<DistributedID,FieldMask>::aligned::iterator finder = 
          global_views.find(did);
        if (finder != global_views.end())
        {
          FieldMask mask;
          derez.deserialize(mask);
          finder->second |= mask;
        }
        else
          derez.deserialize(global_views[did]);
      }
    }

    //--------------------------------------------------------------------------
    void ShardedMappingExchange::initiate_exchange(
                                  const InstanceSet &local_mappings, 
                                  const std::vector<InstanceView*> &local_views)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
        // Populate the data structure with instance names
        for (unsigned idx = 0; idx < local_mappings.size(); idx++)
        {
          const InstanceRef &mapping = local_mappings[idx];
          const FieldMask &mask = mapping.get_valid_fields();
          if (check_mappings)
          {
            const DistributedID did = mapping.get_manager()->did;
            LegionMap<ShardID,FieldMask>::aligned &inst_map = mappings[did];
            LegionMap<ShardID,FieldMask>::aligned::iterator finder = 
              inst_map.find(shard_id);
            if (finder == inst_map.end())
              inst_map[shard_id] = mask;
            else
              finder->second |= mask;
          }
          const DistributedID view_did = local_views[idx]->did;
          LegionMap<DistributedID,FieldMask>::aligned::iterator finder = 
            global_views.find(view_did);
          if (finder == global_views.end())
            global_views[view_did] = mask;
          else
            finder->second |= mask;
        }
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void ShardedMappingExchange::complete_exchange(Operation *op,
                                              ShardedView *sharded_view, 
                                              const InstanceSet &local_mappings,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      if (sharded_view != NULL)
        sharded_view->initialize(global_views, local_mappings, applied_events);
      if (check_mappings)
      {
#ifdef DEBUG_LEGION
        assert(op != NULL);
#endif
        // Check to see if our mappings interfere with any others
        for (unsigned idx = 0; idx < local_mappings.size(); idx++)
        {
          const InstanceRef &mapping = local_mappings[idx];
          const DistributedID did = mapping.get_manager()->did;
          const FieldMask &mask = mapping.get_valid_fields();
          const std::map<DistributedID,
                LegionMap<ShardID,FieldMask>::aligned>::const_iterator
            finder = mappings.find(did);
#ifdef DEBUG_LEGION
          // We should have at least our own
          assert(finder != mappings.end());
#endif
          for (LegionMap<ShardID,FieldMask>::aligned::const_iterator it = 
                finder->second.begin(); it != finder->second.end(); it++)
          {
            // We can skip ourself
            if (it->first == shard_id)
              continue;
            const FieldMask overlap = mask & it->second;
            if (!overlap)
              continue;
            // This is the error condition
            TaskContext *ctx = op->get_context();
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                "%s in control replicated contexts must "
                "map to different instances for the same field. Inline "
                "mapping in shard %d conflicts with mapping in shard %d "
                "of control replciated task %s (UID %lld)", 
                op->get_logging_name(), shard_id, it->first, 
                ctx->get_task_name(), ctx->get_unique_id())
          }
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Template Index Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TemplateIndexExchange::TemplateIndexExchange(ReplicateContext *ctx,
                                                 CollectiveID id)
      : AllGatherCollective(ctx, id), current_stage(-1)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TemplateIndexExchange::TemplateIndexExchange(
                                               const TemplateIndexExchange &rhs)
      : AllGatherCollective(rhs), current_stage(-1)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    TemplateIndexExchange::~TemplateIndexExchange(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TemplateIndexExchange& TemplateIndexExchange::operator=(
                                               const TemplateIndexExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void TemplateIndexExchange::pack_collective_stage(Serializer &rez,int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(index_counts.size());
      for (std::map<int,unsigned>::const_iterator it = 
            index_counts.begin(); it != index_counts.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }
    
    //--------------------------------------------------------------------------
    void TemplateIndexExchange::unpack_collective_stage(Deserializer &derez,
                                                        int stage)
    //--------------------------------------------------------------------------
    {
      // If we are not a participating stage then we already contributed our
      // data into the output so we clear ourself to avoid double counting
      if ((stage == -1) && !participating)
        index_counts.clear();
      size_t num_counts;
      derez.deserialize(num_counts);
      for (unsigned idx = 0; idx < num_counts; idx++)
      {
        int index;
        derez.deserialize(index);
        unsigned count;
        derez.deserialize(count);
        std::map<int,unsigned>::iterator finder = index_counts.find(index);
        if (finder == index_counts.end())
          index_counts[index] = count;
        else
          finder->second += count;
      }
    }

    //--------------------------------------------------------------------------
    void TemplateIndexExchange::initiate_exchange(
                                                const std::vector<int> &indexes)
    //--------------------------------------------------------------------------
    {
      for (std::vector<int>::const_iterator it = indexes.begin();
            it != indexes.end(); it++)
        index_counts[*it] = 1;
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void TemplateIndexExchange::complete_exchange(
                                          std::map<int,unsigned> &result_counts)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait(true/*block*/);
      result_counts.swap(index_counts);
    }

    /////////////////////////////////////////////////////////////
    // Unordered Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    UnorderedExchange::UnorderedExchange(ReplicateContext *ctx, 
                                         CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx), current_stage(-1)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UnorderedExchange::UnorderedExchange(const UnorderedExchange &rhs)
      : AllGatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    UnorderedExchange::~UnorderedExchange(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    UnorderedExchange& UnorderedExchange::operator=(
                                                   const UnorderedExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::update_future_counts(const int stage,
                            std::map<int,std::map<T,unsigned> > &future_counts,
                            std::map<T,unsigned> &counts)
    //--------------------------------------------------------------------------
    {
      typename std::map<int,std::map<T,unsigned> >::iterator next =
        future_counts.find(stage-1);
      if (next != future_counts.end())
      {
        for (typename std::map<T,unsigned>::const_iterator it = 
              next->second.begin(); it != next->second.end(); it++)
        {
          typename std::map<T,unsigned>::iterator finder = 
            counts.find(it->first);
          if (finder == counts.end())
            counts.insert(*it);
          else
            finder->second += it->second;
        }
        future_counts.erase(next);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::pack_counts(Serializer &rez,   
                                        const std::map<T,unsigned> &counts)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(counts.size());
      for (typename std::map<T,unsigned>::const_iterator it = 
            counts.begin(); it != counts.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::unpack_counts(const int stage, Deserializer &derez,   
                                          std::map<T,unsigned> &counts)
    //--------------------------------------------------------------------------
    {
      size_t num_counts;
      derez.deserialize(num_counts);
      if (num_counts == 0)
        return;
      for (unsigned idx = 0; idx < num_counts; idx++)
      {
        T key;
        derez.deserialize(key);
        typename std::map<T,unsigned>::iterator finder = counts.find(key);
        if (finder != counts.end())
        {
          unsigned count;
          derez.deserialize(count);
          finder->second += count;
        }
        else
          derez.deserialize(counts[key]);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::pack_field_counts(Serializer &rez,
                          const std::map<std::pair<T,FieldID>,unsigned> &counts)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(counts.size());
      for (typename std::map<std::pair<T,FieldID>,unsigned>::const_iterator it =
            counts.begin(); it != counts.end(); it++)
      {
        rez.serialize(it->first.first);
        rez.serialize(it->first.second);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void UnorderedExchange::unpack_field_counts(const int stage,
        Deserializer &derez, std::map<std::pair<T,FieldID>,unsigned> &counts)
    //--------------------------------------------------------------------------
    {
      size_t num_counts;
      derez.deserialize(num_counts);
      if (num_counts == 0)
        return;
      for (unsigned idx = 0; idx < num_counts; idx++)
      {
        std::pair<T,FieldID> key;
        derez.deserialize(key.first);
        derez.deserialize(key.second);
        typename std::map<std::pair<T,FieldID>,unsigned>::iterator finder =
          counts.find(key);
        if (finder != counts.end())
        {
          unsigned count;
          derez.deserialize(count);
          finder->second += count;
        }
        else
          derez.deserialize(counts[key]);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T, typename OP>
    void UnorderedExchange::initialize_counts(const std::map<T,OP*> &ops,
                                              std::map<T,unsigned> &counts)
    //--------------------------------------------------------------------------
    {
      for (typename std::map<T,OP*>::const_iterator it = 
            ops.begin(); it != ops.end(); it++) 
        counts[it->first] = 1;
    }

    //--------------------------------------------------------------------------
    template<typename T, typename OP>
    void UnorderedExchange::find_ready_ops(const size_t total_shards,
                const std::map<T,unsigned> &final_counts,
                const std::map<T,OP*> &ops, std::vector<Operation*> &ready_ops)
    //--------------------------------------------------------------------------
    {
      for (typename std::map<T,unsigned>::const_iterator it = 
            final_counts.begin(); it != final_counts.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->second <= total_shards);
#endif
        if (it->second == total_shards)
        {
          typename std::map<T,OP*>::const_iterator finder = ops.find(it->first);
#ifdef DEBUG_LEGION
          assert(finder != ops.end());
#endif
          ready_ops.push_back(finder->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void UnorderedExchange::pack_collective_stage(Serializer &rez, int stage)
    //--------------------------------------------------------------------------
    {
      pack_counts(rez, index_space_counts);
      pack_counts(rez, index_partition_counts);
      pack_counts(rez, field_space_counts);
      pack_field_counts(rez, field_counts);
      pack_counts(rez, logical_region_counts);
      pack_field_counts(rez, detach_counts);
    }

    //--------------------------------------------------------------------------
    void UnorderedExchange::unpack_collective_stage(Deserializer &derez, 
                                                    int stage)
    //--------------------------------------------------------------------------
    {
      // If we are not a participating stage then we already contributed our
      // data into the output so we clear ourself to avoid double counting
      if ((stage == -1) && !participating)
      {
        index_space_counts.clear();
        index_partition_counts.clear();
        field_space_counts.clear();
        field_counts.clear();
        logical_region_counts.clear();
        detach_counts.clear();
      }
      unpack_counts(stage, derez, index_space_counts);
      unpack_counts(stage, derez, index_partition_counts);
      unpack_counts(stage, derez, field_space_counts);
      unpack_field_counts(stage, derez, field_counts);
      unpack_counts(stage, derez, logical_region_counts);
      unpack_field_counts(stage, derez, detach_counts);
    }

    //--------------------------------------------------------------------------
    bool UnorderedExchange::exchange_unordered_ops(
                                    const std::list<Operation*> &unordered_ops,
                                          std::vector<Operation*> &ready_ops)
    //--------------------------------------------------------------------------
    {
      // Sort our operations
      if (!unordered_ops.empty())
      {
        for (std::list<Operation*>::const_iterator it = 
              unordered_ops.begin(); it != unordered_ops.end(); it++)
        {
          switch ((*it)->get_operation_kind())
          {
            case Operation::DELETION_OP_KIND:
              {
#ifdef DEBUG_LEGION
                ReplDeletionOp *op = dynamic_cast<ReplDeletionOp*>(*it);
                assert(op != NULL);
#else
                ReplDeletionOp *op = static_cast<ReplDeletionOp*>(*it);
#endif
                op->record_unordered_kind(index_space_deletions,
                    index_partition_deletions, field_space_deletions,
                    field_deletions, logical_region_deletions); 
                break; 
              }
            case Operation::DETACH_OP_KIND:
              {
#ifdef DEBUG_LEGION
                ReplDetachOp *op = dynamic_cast<ReplDetachOp*>(*it);
                assert(op != NULL);
#else
                ReplDetachOp *op = static_cast<ReplDetachOp*>(*it);
#endif
                op->record_unordered_kind(detachments);
                break;
              }
            default: // Unimplemented operation kind
              assert(false);
          }
        }
        // Set the initial counts to one for all our unordered ops
        initialize_counts(index_space_deletions, index_space_counts);
        initialize_counts(index_partition_deletions, index_partition_counts);
        initialize_counts(field_space_deletions, field_space_counts);
        initialize_counts(field_deletions, field_counts);
        initialize_counts(logical_region_deletions, logical_region_counts);
        initialize_counts(detachments, detach_counts);
      }
      // Perform the exchange
      perform_collective_sync();
      // Now look and see which operations have keys for all shards 
      // Only need to do this if we have ops, if we didn't have ops then
      // it's impossible for anyone else to have them all too
      if (!unordered_ops.empty())
      {
        const size_t total_shards = manager->total_shards;
        find_ready_ops(total_shards, index_space_counts,
                       index_space_deletions, ready_ops);
        find_ready_ops(total_shards, index_partition_counts,
                       index_partition_deletions, ready_ops);
        find_ready_ops(total_shards, field_space_counts,
                       field_space_deletions, ready_ops);
        find_ready_ops(total_shards, field_counts,
                       field_deletions, ready_ops);
        find_ready_ops(total_shards, logical_region_counts,
                       logical_region_deletions, ready_ops);
        find_ready_ops(total_shards, detach_counts,
                       detachments, ready_ops);
      }
      // Return true if anybody anywhere had a non-zero count
      return (!index_space_counts.empty() || !index_partition_counts.empty() ||
          !field_space_counts.empty() || !field_counts.empty() || 
          !logical_region_counts.empty() || !detach_counts.empty());
    }

    /////////////////////////////////////////////////////////////
    // Consensus Match Base 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ConsensusMatchBase::ConsensusMatchBase(ReplicateContext *ctx,
                                           CollectiveIndexLocation loc)
      : AllGatherCollective(loc, ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ConsensusMatchBase::ConsensusMatchBase(const ConsensusMatchBase &rhs)
      : AllGatherCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ConsensusMatchBase::~ConsensusMatchBase(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    /*static*/ void ConsensusMatchBase::handle_consensus_match(const void *args)
    //--------------------------------------------------------------------------
    {
      const ConsensusMatchArgs *margs = (const ConsensusMatchArgs*)args;
      margs->base->complete_exchange();
      delete margs->base;
    }

    /////////////////////////////////////////////////////////////
    // Consensus Match Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    template<typename T>
    ConsensusMatchExchange<T>::ConsensusMatchExchange(ReplicateContext *ctx,
             CollectiveIndexLocation loc, Future f, void *out, ApUserEvent trig)
      : ConsensusMatchBase(ctx, loc), to_complete(f),
        output(static_cast<T*>(out)), to_trigger(trig)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T>
    ConsensusMatchExchange<T>::ConsensusMatchExchange(
                                              const ConsensusMatchExchange &rhs)
      : ConsensusMatchBase(rhs), to_complete(rhs.to_complete),
        output(rhs.output), to_trigger(rhs.to_trigger)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    template<typename T>
    ConsensusMatchExchange<T>::~ConsensusMatchExchange(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    template<typename T>
    ConsensusMatchExchange<T>& ConsensusMatchExchange<T>::operator=(
                                              const ConsensusMatchExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void ConsensusMatchExchange<T>::pack_collective_stage(Serializer &rez, 
                                                          int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(element_counts.size());
      for (typename std::map<T,size_t>::const_iterator it = 
            element_counts.begin(); it != element_counts.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void ConsensusMatchExchange<T>::unpack_collective_stage(
                                                 Deserializer &derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_elements;
      derez.deserialize(num_elements);
      for (unsigned idx = 0; idx < num_elements; idx++)
      {
        T element;
        derez.deserialize(element);
        typename std::map<T,size_t>::iterator finder = 
          element_counts.find(element);
        if (finder != element_counts.end())
        {
          size_t count;
          derez.deserialize(count);
          finder->second += count;
        }
        else
          derez.deserialize(element_counts[element]);
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    bool ConsensusMatchExchange<T>::match_elements_async(const void *input,
                                                         size_t num_elements)
    //--------------------------------------------------------------------------
    {
      const T *inputs = static_cast<const T*>(input);
      for (unsigned idx = 0; idx < num_elements; idx++)
        element_counts[inputs[idx]] = 1;
#ifdef DEBUG_LEGION
      max_elements = num_elements;
#endif
      perform_collective_async(); 
      const RtEvent precondition = perform_collective_wait(false/*block*/);
      if (precondition.exists() && !precondition.has_triggered())
      {
        ConsensusMatchArgs args(this, context->get_unique_id());
        context->runtime->issue_runtime_meta_task(args,
            LG_LATENCY_DEFERRED_PRIORITY, precondition);
        return false;
      }
      else
      {
        complete_exchange();
        return true;
      }
    }

    //--------------------------------------------------------------------------
    template<typename T>
    void ConsensusMatchExchange<T>::complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      const size_t total_shards = manager->total_shards; 
      size_t next_index = 0;
      for (typename std::map<T,size_t>::const_iterator it = 
            element_counts.begin(); it != element_counts.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->second <= total_shards);
#endif
        if (it->second < total_shards)
          continue;
#ifdef DEBUG_LEGION
        assert(next_index < max_elements);
#endif
        output[next_index++] = it->first;
      }
      // A little bit of help from the replicate context to complete the future
      context->help_complete_future(to_complete, &next_index, 
                        sizeof(next_index), false/*own*/);
      Runtime::trigger_event(NULL, to_trigger);
    }

    template class ConsensusMatchExchange<uint8_t>;
    template class ConsensusMatchExchange<uint16_t>;
    template class ConsensusMatchExchange<uint32_t>;
    template class ConsensusMatchExchange<uint64_t>;

    /////////////////////////////////////////////////////////////
    // VerifyReplicableExchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VerifyReplicableExchange::VerifyReplicableExchange(
                             CollectiveIndexLocation loc, ReplicateContext *ctx)
      : AllGatherCollective<false>(loc, ctx)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VerifyReplicableExchange::VerifyReplicableExchange(
                                            const VerifyReplicableExchange &rhs)
      : AllGatherCollective<false>(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VerifyReplicableExchange::~VerifyReplicableExchange(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VerifyReplicableExchange& VerifyReplicableExchange::operator=(
                                            const VerifyReplicableExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void VerifyReplicableExchange::pack_collective_stage(Serializer &rez, 
                                                         int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(unique_hashes.size());
      for (ShardHashes::const_iterator it = unique_hashes.begin();
            it != unique_hashes.end(); it++)
      {
        rez.serialize(it->first.first);
        rez.serialize(it->first.second);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void VerifyReplicableExchange::unpack_collective_stage(Deserializer &derez,
                                                           int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_hashes;
      derez.deserialize(num_hashes);
      for (unsigned idx = 0; idx < num_hashes; idx++)
      {
        std::pair<uint64_t,uint64_t> key;
        derez.deserialize(key.first);
        derez.deserialize(key.second);
        ShardHashes::iterator finder = unique_hashes.find(key);
        if (finder != unique_hashes.end())
        {
          ShardID sid;
          derez.deserialize(sid);
          if (sid < finder->second)
            finder->second = sid;
        }
        else
          derez.deserialize(unique_hashes[key]);
      }
    }

    //--------------------------------------------------------------------------
    const VerifyReplicableExchange::ShardHashes& 
                            VerifyReplicableExchange::exchange(uint64_t hash[2])
    //--------------------------------------------------------------------------
    {
      const std::pair<uint64_t,uint64_t> key(hash[0],hash[1]);
      unique_hashes[key] = local_shard;
      perform_collective_sync();
      return unique_hashes;
    }

    /////////////////////////////////////////////////////////////
    // OutputSizeExchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OutputSizeExchange::OutputSizeExchange(ReplicateContext *ctx,
                                           CollectiveIndexLocation loc,
                                          std::map<unsigned,SizeMap> &all_sizes)
      : AllGatherCollective<false>(loc, ctx), all_output_sizes(all_sizes)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OutputSizeExchange::OutputSizeExchange(const OutputSizeExchange &rhs)
      : AllGatherCollective<false>(rhs), all_output_sizes(rhs.all_output_sizes)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    OutputSizeExchange::~OutputSizeExchange(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    OutputSizeExchange& OutputSizeExchange::operator=(
                                                  const OutputSizeExchange &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void OutputSizeExchange::pack_collective_stage(Serializer &rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize(all_output_sizes.size());
      for (std::map<unsigned,SizeMap>::iterator it = all_output_sizes.begin();
           it != all_output_sizes.end(); ++it)
      {
        rez.serialize(it->first);
        rez.serialize(it->second.size());
        for (SizeMap::iterator sit = it->second.begin();
             sit != it->second.end(); ++sit)
        {
          rez.serialize(sit->first);
          rez.serialize(sit->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void OutputSizeExchange::unpack_collective_stage(
                                                 Deserializer &derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_sizes;
      derez.deserialize(num_sizes);
      if (num_sizes == 0) return;
      for (unsigned idx = 0; idx < num_sizes; ++idx)
      {
        unsigned out_idx;
        derez.deserialize(out_idx);
        SizeMap &sizes = all_output_sizes[out_idx];

        size_t num_entries;
        derez.deserialize(num_entries);
        for (unsigned eidx = 0; eidx < num_entries; eidx++)
        {
          DomainPoint point;
          derez.deserialize(point);
#ifdef DEBUG_LEGION
          size_t size;
          derez.deserialize(size);
          assert(sizes.find(point) == sizes.end() ||
                 sizes.find(point)->second == size);
          sizes[point] = size;
#else
          derez.deserialize(sizes[point]);
#endif
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent OutputSizeExchange::exchange_output_sizes(void)
    //--------------------------------------------------------------------------
    {
      perform_collective_async();
      return perform_collective_wait(false/*block*/);
    }

    /////////////////////////////////////////////////////////////
    // Slow Barrier
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SlowBarrier::SlowBarrier(ReplicateContext *ctx, CollectiveID id)
      : AllGatherCollective<false>(ctx, id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SlowBarrier::SlowBarrier(const SlowBarrier &rhs)
      : AllGatherCollective<false>(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    SlowBarrier::~SlowBarrier(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SlowBarrier& SlowBarrier::operator=(const SlowBarrier &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

  }; // namespace Internal
}; // namespace Legion

