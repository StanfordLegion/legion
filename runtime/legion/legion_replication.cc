/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#include "legion_views.h"
#include "legion_context.h"
#include "legion_replication.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

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
      launch_space = IndexSpace::NO_SPACE;
      versioning_collective_id = UINT_MAX;
      future_collective_id = UINT_MAX;
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
      deactivate_individual_task();
      projection_infos.clear();
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
      // Do the mapper call to get the sharding function to use
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id); 
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      output.chosen_functor = UINT_MAX;
      mapper->invoke_task_select_sharding_functor(this, input, &output);
      if (output.chosen_functor == UINT_MAX)
      {
        log_run.error("Mapper %s failed to pick a valid sharding functor for "
                      "task %s (UID %lld)", mapper->get_mapper_name(),
                      get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
#ifdef DEBUG_LEGION
      assert(sharding_function != NULL);
      // In debug mode we check to make sure that all the mappers
      // picked the same sharding function
      assert(sharding_collective != NULL);
      // Contribute the result
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() && 
          !sharding_collective->validate(this->sharding_functor))
      {
        log_run.error("ERROR: Mapper %s chose different sharding functions "
                      "for individual task %s (UID %lld) in %s "
                      "(UID %lld)", mapper->get_mapper_name(), get_task_name(), 
                      get_unique_id(), parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id());
        assert(false); 
      }
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
      owner_shard = sharding_function->find_owner(index_point, index_domain);
      // If we own it we go on the queue, otherwise we complete early
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        // We don't own it, so we can pretend like we
        // mapped and executed this task already
        // Before we can do that though we have to get the version state
        // names for any writes so we can update our local state
        VersioningInfoBroadcast version_broadcast(repl_ctx, 
                      versioning_collective_id, owner_shard);
        version_broadcast.wait_for_states(map_applied_conditions);
        const UniqueID logical_context_uid = parent_ctx->get_context_uid();
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          if (IS_WRITE(regions[idx]))
          {
            const VersioningSet<> &remote_advance_states = 
              version_broadcast.find_advance_states(idx);
            const RegionRequirement &req = regions[idx];
            const bool parent_is_upper_bound = (req.region == req.parent);
            runtime->forest->advance_remote_versions(this, idx, req,
                parent_is_upper_bound, logical_context_uid, 
                remote_advance_states, map_applied_conditions);
          }
        }
        if (!map_applied_conditions.empty())
          complete_mapping(Runtime::merge_events(map_applied_conditions));
        else
          complete_mapping();
        complete_execution();
        trigger_children_complete();
      }
      else // We own it, so it goes on the ready queue
        enqueue_ready_operation(); 
    }

    //--------------------------------------------------------------------------
    RtEvent ReplIndividualTask::perform_mapping(
                                         MustEpochOp *must_epoch_owner/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      // Do the base call  
      RtEvent result = IndividualTask::perform_mapping(must_epoch_owner);
      // If there is an event then the mapping isn't done so we don't have
      // the final versions yet and can't do the broadcast
      if (result.exists())
        return result;
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Then broadcast the versioning results for any region requirements
      // that are writes which are going to advance the version numbers
      VersioningInfoBroadcast version_broadcast(repl_ctx, 
                    versioning_collective_id, owner_shard);
#ifdef DEBUG_LEGION
      assert(regions.size() == version_infos.size());
#endif
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (IS_WRITE(regions[idx]))
          version_broadcast.pack_advance_states(idx, version_infos[idx]);
      }
      version_broadcast.perform_collective_async();
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::handle_future(const void *res, 
                                           size_t res_size, bool owned)
    //--------------------------------------------------------------------------
    {
      // We have to save the future for locally for when we broadcast it
      if (owned)
      {
        future_store = const_cast<void*>(res);
        future_size = res_size;
      }
      else
      {
        future_size = res_size;
        future_store = legion_malloc(FUTURE_RESULT_ALLOC, future_size);
        memcpy(future_store, res, future_size);
      }
      IndividualTask::handle_future(future_store, future_size, false/*owned*/);
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
      // the future result
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        FutureBroadcast future_collective(repl_ctx, 
                                          future_collective_id, owner_shard);
        future_collective.broadcast_future(future_store, future_size);
      }
      else
      {
        FutureBroadcast future_collective(repl_ctx, 
                                          future_collective_id, owner_shard);
        future_collective.receive_future(result.impl);
      }
      IndividualTask::trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
      versioning_collective_id = ctx->get_next_collective_index();
      future_collective_id = ctx->get_next_collective_index();
      // Also initialize our index domain of a single point
      index_domain = Domain(index_point, index_point);
      launch_space = ctx->find_index_launch_space(index_domain);
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
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
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
      // Do the mapper call to get the sharding function to use
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id); 
      Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
      Mapper::SelectShardingFunctorOutput output;
      output.chosen_functor = UINT_MAX;
      mapper->invoke_task_select_sharding_functor(this, input, &output);
      if (output.chosen_functor == UINT_MAX)
      {
        log_run.error("Mapper %s failed to pick a valid sharding functor for "
                      "task %s (UID %lld)", mapper->get_mapper_name(),
                      get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
#ifdef DEBUG_LEGION
      assert(sharding_function != NULL);
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
      {
        log_run.error("ERROR: Mapper %s chose different sharding functions "
                      "for index task %s (UID %lld) in %s (UID %lld)", 
                      mapper->get_mapper_name(), get_task_name(), 
                      get_unique_id(), parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id());
        assert(false);
      }
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
      const Domain &local_domain = 
        sharding_function->find_shard_domain(repl_ctx->owner_shard->shard_id,
                                             index_domain);
      index_domain = local_domain;
      // If it's empty we're done, otherwise we go back on the queue
      if (local_domain.get_volume() == 0)
      {
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution();
      }
      else // We have valid points, so it goes on the ready queue
        enqueue_ready_operation();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        projection_infos[idx] = 
         ProjectionInfo(runtime, regions[idx], launch_space, sharding_function);
        runtime->forest->perform_dependence_analysis(this, idx, regions[idx], 
                                                     restrict_infos[idx],
                                                     version_infos[idx],
                                                     projection_infos[idx],
                                                     privilege_paths[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      // If we have a reduction operator, exchange the future results
      if (redop > 0)
      {
#ifdef DEBUG_LEGION
        assert(reduction_collective != NULL);
#endif
        // Grab the reduction state buffer and then reinitialize it so
        // that all the shards can be applied to it in the same order 
        // so that we have bit equivalence across the shards
        void *shard_buffer = reduction_state;
        reduction_state = NULL;
        initialize_reduction_state();
        // The collective takes ownership of the buffer here
        reduction_collective->reduce_futures(shard_buffer, this);
      }
      // Then we do the base class thing
      IndexTask::trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_collective == NULL);
      // Check for any non-functional projection functions
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle_type == SINGULAR)
          continue;
        ProjectionFunction *function = 
          runtime->find_projection_function(regions[idx].projection);
        if (!function->is_functional)
        {
          log_run.error("Region requirement %d of task %s (UID %lld) in "
                        "parent task %s (UID %lld) has non-functional "
                        "projection function. All projection functions "
                        "for control replication must be functional.",
                        idx, get_task_name(), get_unique_id(),
                        parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id());
          assert(false);
        }
      }
#endif
      // If we have a reduction op then we need an exchange
      if (redop > 0)
        reduction_collective = new FutureExchange(ctx, reduction_state_size);
    }

    //--------------------------------------------------------------------------
    FutureMapImpl* ReplIndexTask::create_future_map(TaskContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(ctx);
#endif
      // Make a replicate future map 
      return legion_new<ReplFutureMapImpl>(repl_ctx, this, index_domain,runtime,
          runtime->get_available_distributed_id(true/*need continuation*/),
          runtime->address_space);
    }

    /////////////////////////////////////////////////////////////
    // Repl Inter Close Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplInterCloseOp::ReplInterCloseOp(Runtime *rt)
      : InterCloseOp(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplInterCloseOp::ReplInterCloseOp(const ReplInterCloseOp &rhs)
      : InterCloseOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplInterCloseOp::~ReplInterCloseOp(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplInterCloseOp& ReplInterCloseOp::operator=(const ReplInterCloseOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void ReplInterCloseOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_inter_close();
      close_barrier = RtBarrier::NO_RT_BARRIER;
    }

    //--------------------------------------------------------------------------
    void ReplInterCloseOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_inter_close();
      runtime->free_repl_inter_close_op(this);
    }

    //--------------------------------------------------------------------------
    void ReplInterCloseOp::set_close_barrier(RtBarrier close_bar)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!close_barrier.exists());
#endif
      close_barrier = close_bar;
    }

    //--------------------------------------------------------------------------
    void ReplInterCloseOp::post_process_composite_view(CompositeView *view)
    //--------------------------------------------------------------------------
    {
      view->set_shard_invalid_barrier(close_barrier);
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
      {
        log_run.error("Mapper %s failed to pick a valid sharding functor for "
                      "index fill in task %s (UID %lld)", 
                      mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
#ifdef DEBUG_LEGION
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
      {
        log_run.error("ERROR: Mapper %s chose different sharding functions "
                      "for index fill in task %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id());
        assert(false);
      }
#endif
      // Now we can do the normal prepipeline stage
      IndexFillOp::trigger_prepipeline_stage();
    }
    
    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      projection_info = ProjectionInfo(runtime, requirement, 
                                       launch_space, sharding_function);
      runtime->forest->perform_dependence_analysis(this, 0/*idx*/,
                                                   requirement,
                                                   restrict_info,
                                                   version_info,
                                                   projection_info,
                                                   privilege_path);
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Compute the local index space of points for this shard
      const Domain &local_domain = 
        sharding_function->find_shard_domain(repl_ctx->owner_shard->shard_id, 
                                             index_domain);
      index_domain = local_domain;
      // If it's empty we're done, otherwise we go back on the queue
      if (local_domain.get_volume() == 0)
      {
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution();
      }
      else // We have valid points, so it goes on the ready queue
        IndexFillOp::trigger_ready();
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Check for any non-functional projection functions
      if (requirement.handle_type != SINGULAR)
      {
        ProjectionFunction *function = 
          runtime->find_projection_function(requirement.projection);
        if (!function->is_functional)
        {
          log_run.error("Region requirement of index fill op (UID %lld) in "
                        "parent task %s (UID %lld) has non-functional "
                        "projection function. All projection functions "
                        "for control replication must be functional.",
                        get_unique_id(), parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id());
          assert(false);
        }
      }
#endif
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
      versioning_collective_id = ctx->get_next_collective_index();
      // Initialize our index domain of a single point
      index_domain = Domain(index_point, index_point);
      launch_space = ctx->find_index_launch_space(index_domain);
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_copy();
      sharding_functor = UINT_MAX;
      sharding_function = NULL;
      launch_space = IndexSpace::NO_SPACE;
#ifdef DEBUG_LEGION
      sharding_collective = NULL;
#endif
      versioning_collective_id = UINT_MAX;
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      deactivate_copy();
      src_projection_infos.clear();
      dst_projection_infos.clear();
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
      {
        log_run.error("Mapper %s failed to pick a valid sharding functor for "
                      "copy in task %s (UID %lld)", mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
#ifdef DEBUG_LEGION
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
      {
        log_run.error("ERROR: Mapper %s chose different sharding functions "
                      "for copy in task %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(), 
                      parent_ctx->get_unique_id());
        assert(false);
      }
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
      ShardID owner_shard = 
        sharding_function->find_owner(index_point, index_domain); 
      // If we own it we go on the queue, otherwise we complete early
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        // We don't own it, so we can pretend like we
        // mapped and executed this copy already
        // Before we do this though we have to get the version state
        // names for any writes so we can update our local state
        VersioningInfoBroadcast version_broadcast(repl_ctx, 
                      versioning_collective_id, owner_shard);
        version_broadcast.wait_for_states(map_applied_conditions);
        const UniqueID logical_context_uid = parent_ctx->get_context_uid();
        for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
        {
          const VersioningSet<> &remote_advance_states = 
            version_broadcast.find_advance_states(idx);
          RegionRequirement &req = dst_requirements[idx];
          // Switch the privileges to read-write if necessary
          const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
          if (is_reduce_req)
            req.privilege = READ_WRITE;
          const bool parent_is_upper_bound = (req.region == req.parent);
          runtime->forest->advance_remote_versions(this, 
              src_requirements.size() + idx, req,
              parent_is_upper_bound, logical_context_uid, 
              remote_advance_states, map_applied_conditions);
          // Switch the privileges back when we are done
          if (is_reduce_req)
            req.privilege = REDUCE;
        }
        if (!map_applied_conditions.empty())
          complete_mapping(Runtime::merge_events(map_applied_conditions));
        else
          complete_mapping();
        complete_execution();
      }
      else // We own it, so do the base call
      {
        // Do the versioning analysis
        RtEvent ready = perform_local_versioning_analysis();
        // Broadcast the versioning information
        VersioningInfoBroadcast version_broadcast(repl_ctx, 
                      versioning_collective_id, owner_shard);
#ifdef DEBUG_LEGION
        assert(dst_requirements.size() == dst_versions.size());
#endif
        for (unsigned idx = 0; idx < dst_versions.size(); idx++)
          version_broadcast.pack_advance_states(idx, dst_versions[idx]);
        version_broadcast.perform_collective_async();
        // Then we can do the enqueue
        enqueue_ready_operation(ready);
      }
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
      {
        log_run.error("Mapper %s failed to pick a valid sharding functor for "
                      "index copy in task %s (UID %lld)", 
                      mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      this->sharding_functor = output.chosen_functor;
      sharding_function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor); 
#ifdef DEBUG_LEGION
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
      {
        log_run.error("ERROR: Mapper %s chose different sharding functions "
                      "for index copy in task %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id());
        assert(false);
      }
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
        src_projection_infos[idx] = 
          ProjectionInfo(runtime, src_requirements[idx], 
                         launch_space, sharding_function);
        runtime->forest->perform_dependence_analysis(this, idx, 
                                                     src_requirements[idx],
                                                     src_restrict_infos[idx],
                                                     src_versions[idx],
                                                     src_projection_infos[idx],
                                                     src_privilege_paths[idx]);
      }
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        dst_projection_infos[idx] = 
          ProjectionInfo(runtime, dst_requirements[idx], 
                         launch_space, sharding_function);
        unsigned index = src_requirements.size()+idx;
        // Perform this dependence analysis as if it was READ_WRITE
        // so that we can get the version numbers correct
        const bool is_reduce_req = IS_REDUCE(dst_requirements[idx]);
        if (is_reduce_req)
          dst_requirements[idx].privilege = READ_WRITE;
        runtime->forest->perform_dependence_analysis(this, index, 
                                                     dst_requirements[idx],
                                                     dst_restrict_infos[idx],
                                                     dst_versions[idx],
                                                     dst_projection_infos[idx],
                                                     dst_privilege_paths[idx]);
        // Switch the privileges back when we are done
        if (is_reduce_req)
          dst_requirements[idx].privilege = REDUCE;
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
      const Domain &local_domain = 
        sharding_function->find_shard_domain(repl_ctx->owner_shard->shard_id,
                                             index_domain);
      index_domain = local_domain;
      // If it's empty we're done, otherwise we go back on the queue
      if (local_domain.get_volume() == 0)
      {
        // We have no local points, so we can just trigger
        complete_mapping();
        complete_execution();
      }
      else // If we have any valid points do the base call
        IndexCopyOp::trigger_ready();
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::initialize_replication(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      for (unsigned idx = 0; idx < dst_requirements.size(); idx++)
      {
        if (dst_requirements[idx].handle_type == SINGULAR)
          continue;
        ProjectionFunction *function = 
          runtime->find_projection_function(dst_requirements[idx].projection);
        if (!function->is_functional)
        {
          log_run.error("Destination region requirement %d of index copy "
                        "(UID %lld) in parent task %s (UID %lld) has "
                        "non-functional projection function. All projection "
                        "functions for control replication must be functional.",
                        idx, get_unique_id(), parent_ctx->get_task_name(), 
                        parent_ctx->get_unique_id());
          assert(false);
        }
      }
#endif
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
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Shard 0 will hold all the deletions
      if (repl_ctx->owner_shard->shard_id == 0)
      {
        // We don't own it, so we can pretend like we
        // mapped and executed this deletion already 
        complete_mapping();
        complete_execution();
      }
      else // We own it, so enqueue it
        enqueue_ready_operation();
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
    void ReplPendingPartitionOp::trigger_mapping(void)
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
      ApEvent ready_event = thunk->perform_shard(this, runtime->forest,
        repl_ctx->owner_shard->shard_id, repl_ctx->shard_manager->total_shards);
      complete_mapping();
      Runtime::trigger_event(completion_event, ready_event);
      need_completion_trigger = false;
      complete_execution(Runtime::protect_event(ready_event));
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
                                                       ApEvent ready_event,
                                                       IndexPartition pid,
                                                       LogicalRegion handle, 
                                                       LogicalRegion parent,
                                                       FieldID fid,
                                                       MapperID id, 
                                                       MappingTagID t)
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
      requirement = RegionRequirement(handle, READ_ONLY, EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ReplByFieldThunk(ctx, pid);
      partition_ready = ready_event;
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_image(ReplicateContext *ctx, 
                                                       ShardID target_shard,
                                                       ApEvent ready_event,
                                                       IndexPartition pid,
                                                   LogicalPartition projection,
                                             LogicalRegion parent, FieldID fid,
                                                   MapperID id, MappingTagID t) 
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
      requirement = RegionRequirement(proj_parent, READ_ONLY, EXCLUSIVE,parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ReplByImageThunk(ctx, target_shard, 
                                   pid, projection.get_index_partition());
      partition_ready = ready_event;
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_image_range(
                                                         ReplicateContext *ctx, 
                                                         ShardID target_shard,
                                                         ApEvent ready_event,
                                                         IndexPartition pid,
                                                LogicalPartition projection,
                                                LogicalRegion parent,
                                                FieldID fid, MapperID id,
                                                MappingTagID t) 
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
      requirement = RegionRequirement(proj_parent, READ_ONLY, EXCLUSIVE,parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ReplByImageRangeThunk(ctx, target_shard, 
                                        pid, projection.get_index_partition());
      partition_ready = ready_event;
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_preimage(ReplicateContext *ctx,
                                    ShardID target_shard, ApEvent ready_event,
                                    IndexPartition pid, IndexPartition proj,
                                    LogicalRegion handle, LogicalRegion parent,
                                    FieldID fid, MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(pid,
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
      requirement = RegionRequirement(handle, READ_ONLY, EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ReplByPreimageThunk(ctx, target_shard, pid, proj);
      partition_ready = ready_event;
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::initialize_by_preimage_range(
                                    ReplicateContext *ctx, ShardID target_shard,
                                    ApEvent ready_event,
                                    IndexPartition pid, IndexPartition proj,
                                    LogicalRegion handle, LogicalRegion parent,
                                    FieldID fid, MapperID id, MappingTagID t)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      if (!runtime->forest->check_partition_by_field_size(pid,
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
      requirement = RegionRequirement(handle, READ_ONLY, EXCLUSIVE, parent);
      requirement.add_field(fid);
      map_id = id;
      tag = t;
#ifdef DEBUG_LEGION
      assert(thunk == NULL);
#endif
      thunk = new ReplByPreimageRangeThunk(ctx, target_shard, pid, proj);
      partition_ready = ready_event;
      if (Runtime::legion_spy_enabled)
        perform_logging();
    }

    //--------------------------------------------------------------------------
    void ReplDependentPartitionOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_dependent_op();
      sharding_functor = UINT_MAX;
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
    void ReplDependentPartitionOp::trigger_prepipeline_stage(void)
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
      mapper->invoke_partition_select_sharding_functor(this, input, &output);
      if (output.chosen_functor == UINT_MAX)
      {
        log_run.error("Mapper %s failed to pick a valid sharding functor for "
                      "dependent partition in task %s (UID %lld)", 
                      mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      this->sharding_functor = output.chosen_functor;
#ifdef DEBUG_LEGION
      assert(sharding_collective != NULL);
      sharding_collective->contribute(this->sharding_functor);
      if (sharding_collective->is_target() &&
          !sharding_collective->validate(this->sharding_functor))
      {
        log_run.error("ERROR: Mapper %s chose different sharding functions "
                      "for dependent partition op in task %s (UID %lld)", 
                      mapper->get_mapper_name(), parent_ctx->get_task_name(),
                      parent_ctx->get_unique_id());
        assert(false);
      }
#endif
      // Now we can do the normal prepipeline stage
      DependentPartitionOp::trigger_prepipeline_stage();
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
      // Get the sharding function implementation to use from our context
      ShardingFunction *function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      // Do different things if this is an index space point or a single point
      if (is_index_space)
      {
        // Compute the local index space of points for this shard
        const Domain &local_domain = 
          function->find_shard_domain(repl_ctx->owner_shard->shard_id, 
                                      index_domain);
        index_domain = local_domain;
        // If it's empty we're done, otherwise we go back on the queue
        if (local_domain.get_volume() == 0)
        {
          // We have no local points, so we can just trigger
          complete_mapping();
          complete_execution();
        }
        else // If we have valid points then we do the base call
          DependentPartitionOp::trigger_ready();
      }
      else
      {
        // Figure out whether this shard owns this point
        ShardID owner_shard = function->find_owner(index_point, index_domain); 
        // If we own it we go on the queue, otherwise we complete early
        if (owner_shard != repl_ctx->owner_shard->shard_id)
        {
          // We don't own it, so we can pretend like we
          // mapped and executed this task already
          complete_mapping();
          complete_execution();
        }
        else // If we're the shard then we do the base call
          DependentPartitionOp::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplByFieldThunk::ReplByFieldThunk(
                                        ReplicateContext *ctx, IndexPartition p)
      : ByFieldThunk(p), collective(FieldDescriptorExchange(ctx))
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
        // Do the all-to-all gather of the field data descriptors
        ApEvent all_ready = collective.exchange_descriptors(instances_ready,
                                                            instances);
        return forest->create_partition_by_field(op, pid, 
                                    collective.descriptors, all_ready); 
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_field(op, pid, 
                                                 instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplByImageThunk::ReplByImageThunk(
                                          ReplicateContext *ctx, ShardID target,
                                          IndexPartition p, IndexPartition proj)
      : ByImageThunk(p, proj), 
        gather_collective(FieldDescriptorGather(ctx, target))
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
        gather_collective.contribute(instances_ready, instances);
        if (gather_collective.is_target())
        {
          ApEvent all_ready;
          const std::vector<FieldDataDescriptor> &full_descriptors =
            gather_collective.get_full_descriptors(all_ready);
          // Perform the operation
          return forest->create_partition_by_image(op, pid, projection, 
                                          full_descriptors, all_ready);
        }
        else // nothing else for us to do
          return ApEvent::NO_AP_EVENT;
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_image(op, pid, projection, 
                                                 instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplByImageRangeThunk::ReplByImageRangeThunk(
                                          ReplicateContext *ctx, ShardID target,
                                          IndexPartition p, IndexPartition proj)
      : ByImageRangeThunk(p, proj), 
        gather_collective(FieldDescriptorGather(ctx, target))
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
        gather_collective.contribute(instances_ready, instances);
        if (gather_collective.is_target())
        {
          ApEvent all_ready;
          const std::vector<FieldDataDescriptor> &full_descriptors =
            gather_collective.get_full_descriptors(all_ready);
          // Perform the operation
          return forest->create_partition_by_image_range(op, pid, projection,
                                                full_descriptors, all_ready);
        }
        else // nothing else for us to do
          return ApEvent::NO_AP_EVENT;
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_image(op, pid, projection, 
                                                 instances, instances_ready);
    }

    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplByPreimageThunk::ReplByPreimageThunk(
                                          ReplicateContext *ctx, ShardID target,
                                          IndexPartition p, IndexPartition proj)
      : ByPreimageThunk(p, proj), 
        gather_collective(FieldDescriptorGather(ctx, target))
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
          return forest->create_partition_by_preimage(op, pid, projection,
                                              full_descriptors, all_ready);
        }
        else // nothing else for us to do
          return ApEvent::NO_AP_EVENT;
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_image(op, pid, projection, 
                                                 instances, instances_ready);
    }
    
    //--------------------------------------------------------------------------
    ReplDependentPartitionOp::ReplByPreimageRangeThunk::
                 ReplByPreimageRangeThunk(ReplicateContext *ctx, ShardID target,
                                          IndexPartition p, IndexPartition proj)
      : ByPreimageRangeThunk(p, proj), 
        gather_collective(FieldDescriptorGather(ctx, target))
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
          return forest->create_partition_by_preimage_range(op, pid, projection,
                                                   full_descriptors, all_ready);
        }
        else // nothing else for us to do
          return ApEvent::NO_AP_EVENT;
      }
      else // singular so just do the normal thing
        return forest->create_partition_by_image(op, pid, projection, 
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
      index_domain = Domain::NO_DOMAIN;
      broadcast = NULL;
      exchange = NULL;
#ifdef DEBUG_LEGION
      sharding_collective = NULL;
#endif
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
      deactivate_must_epoch_op();
      if (broadcast != NULL)
        delete broadcast;
      if (exchange != NULL)
        delete exchange;
#ifdef DEBUG_LEGION
      if (sharding_collective != NULL)
        delete sharding_collective;
#endif
      runtime->free_repl_epoch_op(this);
    }

    //--------------------------------------------------------------------------
    FutureMapImpl* ReplMustEpochOp::create_future_map(TaskContext *ctx,
                                                        IndexSpace launch_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(launch_space.exists());
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(ctx);
#endif
      runtime->forest->find_launch_space_domain(launch_space, index_domain);
      return legion_new<ReplFutureMapImpl>(repl_ctx, this, index_domain,runtime,
          runtime->get_available_distributed_id(true/*need continuation*/),
          runtime->address_space);
    }

    //--------------------------------------------------------------------------
    MapperManager* ReplMustEpochOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      ReplicateContext *repl_ctx = dynamic_cast<ReplicateContext*>(parent_ctx);
      assert(repl_ctx != NULL);
#else
      ReplicateContext *repl_ctx = static_cast<ReplicateContext*>(parent_ctx);
#endif
      // Fill in the shard map so that we get the sharding ID
      input.shard_mapping = repl_ctx->shard_manager->shard_mapping; 
      output.chosen_functor = UINT_MAX;
      // Shard the constraints so that each mapper call handles 
      // a subset of the constraints when performing the mapping 
      std::vector<Mapper::MappingConstraint> local_constraints;
      for (unsigned idx = repl_ctx->owner_shard->shard_id; 
            idx < input.constraints.size(); 
            idx += repl_ctx->shard_manager->total_shards)
        local_constraints.push_back(input.constraints[idx]);
      const size_t total_constraints = input.constraints.size();
      input.constraints = local_constraints;
      // Do the mapper call
      Processor mapper_proc = parent_ctx->get_executing_processor();
      MapperManager *mapper = runtime->find_mapper(mapper_proc, mapper_id);
      // We've got all our meta-data set up so go ahead and issue the call
      mapper->invoke_map_must_epoch(this, &input, &output);
      // Check that we have a sharding ID
      if (output.chosen_functor == UINT_MAX)
      {
        log_run.error("Invalid mapper output from invocation of "
            "'map_must_epoch' on mapper %s. Mapper failed to specify "
            "a valid sharding ID for a must epoch operation in control "
            "replicated context of task %s (UID %lld).",
            mapper->get_mapper_name(), repl_ctx->get_task_name(),
            repl_ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      sharding_functor = output.chosen_functor;
#ifdef DEBUG_LEGION
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
      assert(broadcast != NULL);
      assert(exchange != NULL);
      assert(result_map.impl != NULL);
      ReplFutureMapImpl *impl = 
          dynamic_cast<ReplFutureMapImpl*>(result_map.impl);
      assert(impl != NULL);
#else
      ReplFutureMapImpl *impl = 
          static_cast<ReplFutureMapImpl*>(result_map.impl);
#endif
      // Set the future map sharding functor
      ShardingFunction *sharding_function = 
          repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      impl->set_sharding_function(sharding_function);
      // Broadcast the processor decisions from shard 0
      // so we can check that they are all the same
      if (repl_ctx->owner_shard->shard_id == 0)
        broadcast->broadcast_processors(output.task_processors);
      // Exchange the constraint mappings so that all ops have all the mappings
      exchange->exchange_must_epoch_mappings(repl_ctx->owner_shard->shard_id,
          repl_ctx->shard_manager->total_shards, total_constraints,
          output.constraint_mappings);
      // Receive processor decisions from shard 0
      if ((repl_ctx->owner_shard->shard_id != 0) &&
          !broadcast->validate_processors(output.task_processors))
      {
        log_run.error("ERROR: Mapper %s chose different processor mappings "
                      "for 'map_must_epoch' call across different shards in "
                      "task %s (UID %lld).", mapper->get_mapper_name(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      // Last we need to prune out any tasks which aren't local to our shard
      std::vector<SingleTask*> local_single_tasks;
      for (std::vector<SingleTask*>::const_iterator it = single_tasks.begin();
            it != single_tasks.end(); it++)
      {
        // Figure out which shard this point belongs to
        ShardID shard = 
          sharding_function->find_owner((*it)->index_point, index_domain);
        // If it's local we can keep going
        if (shard == repl_ctx->owner_shard->shard_id)
          continue;
        // Otherwise we need to make it look like it is already done
        // TODO: Figure out how to make our must epoch operation only
        // run the points for our local shard
        assert(false);
      }
      return mapper;
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::initialize_collectives(ReplicateContext *ctx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(broadcast == NULL);
      assert(exchange == NULL);
#endif
      broadcast = new MustEpochProcessorBroadcast(ctx, 0/*owner shard*/);
      exchange = new MustEpochMappingExchange(ctx);
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
      // Shard 0 will handle the timing operation
      if (repl_ctx->owner_shard->shard_id > 0)
      {
        complete_mapping();
        // Trigger this when the timing barrier is done
        DeferredExecuteArgs args;
        args.proxy_this = this;
        runtime->issue_runtime_meta_task(args, LG_LATENCY_PRIORITY, this, 
                                         timing_collective->get_done_event());
      }
      else // Shard 0 does the normal timing operation
        Operation::trigger_mapping();
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
        long long value = *timing_collective;
        result.impl->set_result(&value, sizeof(value), false);
      }
      else
      {
        // Perform the measurement and then arrive on the barrier
        // with the result to broadcast it to the other shards
        switch (measurement)
        {
          case MEASURE_SECONDS:
            {
              double value = Realm::Clock::current_time();
              result.impl->set_result(&value, sizeof(value), false);
              long long *ptr = reinterpret_cast<long long*>(&value);
              timing_collective->broadcast(*ptr);
              break;
            }
          case MEASURE_MICRO_SECONDS:
            {
              long long value = Realm::Clock::current_time_in_microseconds();
              result.impl->set_result(&value, sizeof(value), false);
              timing_collective->broadcast(value);
              break;
            }
          case MEASURE_NANO_SECONDS:
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
      complete_execution();
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

    /////////////////////////////////////////////////////////////
    // Shard Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardManager::ShardManager(Runtime *rt, ControlReplicationID id, 
        size_t total, unsigned index, AddressSpaceID owner,SingleTask *original)
      : runtime(rt), repl_id(id), total_shards(total), 
        address_space_index(index),owner_space(owner), original_task(original),
        manager_lock(Reservation::create_reservation()), address_spaces(NULL),
        local_mapping_complete(0), remote_mapping_complete(0),
        trigger_local_complete(0), trigger_remote_complete(0),
        trigger_local_commit(0), trigger_remote_commit(0), 
        remote_constituents(0), first_future(true) 
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(total_shards > 0);
#endif
      runtime->register_shard_manager(repl_id, this);
      if (owner_space == runtime->address_space)
      {
        pending_partition_barrier = 
          ApBarrier(Realm::Barrier::create_barrier(total_shards));
        future_map_barrier = 
          ApBarrier(Realm::Barrier::create_barrier(total_shards));
      }
    }

    //--------------------------------------------------------------------------
    ShardManager::ShardManager(const ShardManager &rhs)
      : runtime(NULL), repl_id(0), total_shards(0), address_space_index(0), 
        owner_space(0), original_task(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }
    
    //--------------------------------------------------------------------------
    ShardManager::~ShardManager(void)
    //--------------------------------------------------------------------------
    { 
      if ((address_spaces != NULL) && address_spaces->remove_reference())
        delete address_spaces;
      // We can delete our shard tasks
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
        delete (*it);
      local_shards.clear();
      // Finally unregister ourselves with the runtime
      const bool owner_manager = (owner_space == runtime->address_space);
      runtime->unregister_shard_manager(repl_id, owner_manager);
      manager_lock.destroy_reservation();
      manager_lock = Reservation::NO_RESERVATION;
      if (owner_manager)
      {
        pending_partition_barrier.destroy_barrier();
        future_map_barrier.destroy_barrier();
      }
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
    void ShardManager::launch(const std::vector<AddressSpaceID> &spaces,
                              const std::map<ShardID,Processor> &mapping)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(original_task != NULL); // should only be called on the owner
      assert(address_spaces == NULL);
#endif
      address_spaces = new ShardMapping(spaces);
      address_spaces->add_reference();
      shard_mapping = mapping;
      // Make our local shards
      create_shards();
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
        (*it)->clone_single_from(original_task);
      // Recursively spawn any other tasks across the machine
      if (address_spaces->size() > 1)
      {
        RtUserEvent ready_event = Runtime::create_rt_user_event();
        broadcast_launch(ready_event, ready_event, original_task);
        // Spawn a task to launch the tasks when ready
        ShardManagerLaunchArgs args;
        args.manager = this;
        runtime->issue_runtime_meta_task(args, LG_LATENCY_PRIORITY, 
                                         original_task, ready_event);
      }
      else
        launch_shards();
    }

    //--------------------------------------------------------------------------
    void ShardManager::unpack_launch(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      RtEvent ready_event;
      derez.deserialize(ready_event);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      // Unpack our local information
      size_t num_procs;
      derez.deserialize(num_procs);
      for (unsigned idx = 0; idx < num_procs; idx++)
      {
        ShardID shard_id;
        derez.deserialize(shard_id);
        derez.deserialize(shard_mapping[shard_id]);
      }
      size_t num_spaces;
      derez.deserialize(num_spaces);
#ifdef DEBUG_LEGION
      assert(address_spaces == NULL);
#endif
      address_spaces = new ShardMapping();
      address_spaces->add_reference();
      address_spaces->resize(num_spaces);
      for (unsigned idx = 0; idx < num_spaces; idx++)
        derez.deserialize((*address_spaces)[idx]);
      derez.deserialize(pending_partition_barrier);
      derez.deserialize(future_map_barrier);
      // Unpack our first shard here
      create_shards();
      ShardTask *first_shard = local_shards[0];
      RtEvent shard_ready = first_shard->unpack_shard_task(derez);
      // Check to see if this shard is ready or not
      // If not build a continuation to avoid blocking the virtual channel
      if (!shard_ready.has_triggered())
      {
        ShardManagerCloneArgs args;
        args.manager = this;
        args.ready_event = ready_event;
        args.to_trigger = to_trigger;
        args.first_shard = first_shard;
        runtime->issue_runtime_meta_task(args, LG_LATENCY_PRIORITY,
                                         first_shard, shard_ready);
      }
      else
        clone_and_launch(ready_event, to_trigger, first_shard);
    }
      
    //--------------------------------------------------------------------------
    void ShardManager::clone_and_launch(RtEvent ready_event,
                                 RtUserEvent to_trigger, ShardTask *first_shard)
    //--------------------------------------------------------------------------
    {
      // Broadcast the launch to the next nodes
      broadcast_launch(ready_event, to_trigger, first_shard);
      // Clone points for all our local shards
      if (local_shards.size() > 1)
      {
        for (std::vector<ShardTask*>::const_iterator it = 
              local_shards.begin(); it != local_shards.end(); it++)
        {
          if ((*it) == first_shard)
            continue;
          // Clone the necessary meta-data
          (*it)->clone_single_from(first_shard);
        }
      }
      // Perform our launches
      if (!ready_event.has_triggered())
      {
        // Spawn a task to launch the tasks when ready
        ShardManagerLaunchArgs args;
        args.manager = this;
        runtime->issue_runtime_meta_task(args, LG_LATENCY_PRIORITY,
                                         first_shard, ready_event);
      }
      else
        launch_shards();
    }

    //--------------------------------------------------------------------------
    void ShardManager::create_shards(void)
    //--------------------------------------------------------------------------
    {
      // Iterate through and find the shards that we have locally 
      for (std::map<ShardID,Processor>::const_iterator it = 
            shard_mapping.begin(); it != shard_mapping.end(); it++)
      {
        AddressSpaceID space = it->second.address_space();
        if (space != runtime->address_space)
          continue;
        local_shards.push_back(
            new ShardTask(runtime, this, it->first, it->second) );
      }
#ifdef DEBUG_LEGION
      assert(!local_shards.empty()); // better have made some shards
#endif
    }

    //--------------------------------------------------------------------------
    void ShardManager::launch_shards(void) const
    //--------------------------------------------------------------------------
    {
      for (std::vector<ShardTask*>::const_iterator it = 
            local_shards.begin(); it != local_shards.end(); it++)
      {
        // If it is a leaf and has no virtual instances then we can mark
        // it mapped right now, otherwise wait for the call back
        if ((*it)->is_leaf() && !(*it)->has_virtual_instances())
          (*it)->complete_mapping();
        // Speculation can always be resolved here
        (*it)->resolve_speculation();
        // Then launch the task for execution
        (*it)->launch_task();
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::broadcast_launch(RtEvent ready_event,
                                   RtUserEvent to_trigger, SingleTask *to_clone)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert((*address_spaces)[address_space_index] == runtime->address_space);
#endif
      std::set<RtEvent> preconditions;
      const unsigned phase_offset = 
        (address_space_index+1) * Runtime::legion_collective_radix;
      for (int idx = 0; idx < Runtime::legion_collective_radix; idx++)
      {
        unsigned index = phase_offset + idx - 1;
        if (index >= address_spaces->size())
          break;
        // Update the number of remote constituents
        remote_constituents++;
        RtUserEvent done = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          // Package up the information we need to send to the next manager
          rez.serialize(repl_id);
          rez.serialize(total_shards);
          rez.serialize(index);
          rez.serialize(ready_event);
          rez.serialize(done);
          rez.serialize<size_t>(shard_mapping.size());
          for (std::map<ShardID,Processor>::const_iterator it = 
                shard_mapping.begin(); it != shard_mapping.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
          rez.serialize<size_t>(address_spaces->size());
          for (unsigned idx = 0; idx < address_spaces->size(); idx++)
            rez.serialize((*address_spaces)[idx]);   
          rez.serialize(pending_partition_barrier);
          rez.serialize(future_map_barrier);
          to_clone->pack_as_shard_task(rez, (*address_spaces)[index]); 
        }
        // Send the message
        runtime->send_control_rep_launch((*address_spaces)[index], rez);
        // Add the event to the preconditions
        preconditions.insert(done);
      }
      if (!preconditions.empty())
        Runtime::trigger_event(to_trigger,Runtime::merge_events(preconditions));
      else
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    bool ShardManager::broadcast_delete(RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      // Send messages to any constituents
      std::set<RtEvent> preconditions;
      const unsigned phase_offset = 
        (address_space_index+1) * Runtime::legion_collective_radix;
      for (int idx = 0; idx < Runtime::legion_collective_radix; idx++)
      {
        unsigned index = phase_offset + idx - 1;
        if (index >= address_spaces->size())
          break;
        RtUserEvent done = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(repl_id);
          rez.serialize(done);
        }
        runtime->send_control_rep_delete((*address_spaces)[index], rez);
        preconditions.insert(done);
      }
      if (!preconditions.empty())
      {
        // Launch a task to perform the deletion when it is ready
        ShardManagerDeleteArgs args;
        args.manager = this;
        RtEvent precondition = 
         runtime->issue_runtime_meta_task(args, LG_LATENCY_PRIORITY, NULL, 
                                          Runtime::merge_events(preconditions));
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger, precondition);
        return false;
      }
      else
      {
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
        return true;
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_post_mapped(bool local)
    //--------------------------------------------------------------------------
    {
      bool notify = false;   
      {
        AutoLock m_lock(manager_lock);
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
        if (original_task == NULL)
        {
          Serializer rez;
          rez.serialize(repl_id);
          runtime->send_control_rep_post_mapped(owner_space, rez);
        }
        else
          original_task->handle_post_mapped(RtEvent::NO_RT_EVENT);
      }
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_future(const void *res,size_t res_size,bool owned)
    //--------------------------------------------------------------------------
    {
      bool notify = false;
      {
        AutoLock m_lock(manager_lock);
        notify = first_future;
        first_future = false;
      }
      if (notify && (original_task != NULL))
        original_task->handle_future(res, res_size, owned);
      else if (owned) // if we own it and don't use it we need to free it
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
          runtime->send_control_rep_trigger_complete(owner_space, rez);
        }
        else
        {
          // Return the privileges first if this isn't the top-level task
          if (!original_task->is_top_level_task())
            local_shards[0]->return_privilege_state(
                              original_task->get_context());
          original_task->trigger_task_complete();
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
          runtime->send_control_rep_trigger_commit(owner_space, rez);
        }
        else
          original_task->trigger_task_commit();
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
        DerezCheck z(derez);
        // Have to unpack the preample we already know
        ControlReplicationID local_repl;
        derez.deserialize(local_repl);
        handle_collective_message(derez);
      }
      else
        runtime->send_control_rep_collective_message(target_space, rez);
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
        DerezCheck z(derez);
        // Have to unpack the preample we already know
        ControlReplicationID local_repl;
        derez.deserialize(local_repl);     
        handle_future_map_request(derez);
      }
      else
        runtime->send_repl_future_map_request(target_space, rez);
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
    void ShardManager::send_composite_view_request(ShardID target, 
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
        DerezCheck z(derez);
        // Have to unpack the preample we already know
        ControlReplicationID local_repl;
        derez.deserialize(local_repl);
        handle_composite_view_request(derez);
      }
      else
        runtime->send_repl_composite_view_request(target_space, rez);
    }

    //--------------------------------------------------------------------------
    void ShardManager::handle_composite_view_request(Deserializer &derez)
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
          (*it)->handle_composite_view_request(derez);
          return;
        }
      }
      // Should never get here
      assert(false);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_clone(const void *args)
    //--------------------------------------------------------------------------
    {
      const ShardManagerCloneArgs *cargs = (const ShardManagerCloneArgs*)args;
      cargs->manager->clone_and_launch(cargs->ready_event, cargs->to_trigger,
                                       cargs->first_shard);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_launch(const void *args)
    //--------------------------------------------------------------------------
    {
      const ShardManagerLaunchArgs *largs = (const ShardManagerLaunchArgs*)args;
      largs->manager->launch_shards();
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_delete(const void *args)
    //--------------------------------------------------------------------------
    {
      const ShardManagerDeleteArgs *dargs = (const ShardManagerDeleteArgs*)args;
      delete dargs->manager;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_launch(Deserializer &derez, 
                                        Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ControlReplicationID repl_id;
      derez.deserialize(repl_id);
      size_t total_shards;
      derez.deserialize(total_shards);
      int index;
      derez.deserialize(index);
      ShardManager *manager = 
        new ShardManager(runtime, repl_id, total_shards, index, source);
      manager->unpack_launch(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_delete(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ControlReplicationID repl_id;
      derez.deserialize(repl_id);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      if (manager->broadcast_delete(to_trigger))
        delete manager;
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_post_mapped(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ControlReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_post_mapped(false/*local*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_trigger_complete(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ControlReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->trigger_task_complete(false/*local*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_trigger_commit(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      ControlReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->trigger_task_commit(false/*local*/);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_collective_message(Deserializer &derez,
                                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ControlReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_collective_message(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_future_map_request(Deserializer &derez,
                                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ControlReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_future_map_request(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ShardManager::handle_composite_view_request(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      ControlReplicationID repl_id;
      derez.deserialize(repl_id);
      ShardManager *manager = runtime->find_shard_manager(repl_id);
      manager->handle_composite_view_request(derez);
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
        new ShardingFunction(functor, sid, total_shards-1);
      // Save the result for the future
      sharding_functions[sid] = result;
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Shard Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardCollective::ShardCollective(ReplicateContext *ctx)
      : manager(ctx->shard_manager), context(ctx), 
        local_shard(ctx->owner_shard->shard_id), 
        collective_index(ctx->get_next_collective_index()),
        collective_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
      // Register this with the context
      context->register_collective(this);
    }

    //--------------------------------------------------------------------------
    ShardCollective::ShardCollective(ReplicateContext *ctx, CollectiveID id)
      : manager(ctx->shard_manager), context(ctx), 
        local_shard(ctx->owner_shard->shard_id), collective_index(id),
        collective_lock(Reservation::create_reservation())
    //--------------------------------------------------------------------------
    {
      // Register this with the context
      context->register_collective(this);
    }

    //--------------------------------------------------------------------------
    ShardCollective::~ShardCollective(void)
    //--------------------------------------------------------------------------
    {
      // Unregister this with the context 
      context->unregister_collective(this);
      collective_lock.destroy_reservation();
      collective_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    int ShardCollective::convert_to_index(ShardID id, ShardID origin) const
    //--------------------------------------------------------------------------
    {
      // shift everything so that the target shard is at index 0 and then add 1
      const int result = 
        ((id + (manager->total_shards - origin)) % manager->total_shards) + 1;
      return result;
    }

    //--------------------------------------------------------------------------
    ShardID ShardCollective::convert_to_shard(int index, ShardID origin) const
    //--------------------------------------------------------------------------
    {
      // shift back to zero indexing and add target then take the modulus
      const ShardID result = (index + origin - 1) % manager->total_shards; 
      return result;
    }

    /////////////////////////////////////////////////////////////
    // Gather Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BroadcastCollective::BroadcastCollective(ReplicateContext *ctx, ShardID o)
      : ShardCollective(ctx), origin(o),
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
    void BroadcastCollective::perform_collective_async(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_shard == origin);
#endif
      send_messages(); 
    }

    //--------------------------------------------------------------------------
    void BroadcastCollective::perform_collective_wait(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_shard != origin);
#endif     
      if (!done_event.has_triggered())
        done_event.lg_wait();
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
      for (int idx = 0; idx < shard_collective_radix; idx++)
      {
        const int target_index = local_index * shard_collective_radix + idx; 
        if (target_index > manager->total_shards)
          break;
        ShardID target = convert_to_shard(target_index, origin);
        Serializer rez;
        {
          RezCheck z(rez);
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
    GatherCollective::GatherCollective(ReplicateContext *ctx, ShardID t)
      : ShardCollective(ctx), target(t), 
        shard_collective_radix(ctx->get_shard_collective_radix()),
        expected_notifications(compute_expected_notifications()),
        received_notifications(0)
    //--------------------------------------------------------------------------
    {
      if (local_shard == target)
        done_event = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    GatherCollective::~GatherCollective(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void GatherCollective::perform_collective_async(void)
    //--------------------------------------------------------------------------
    {
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
        if (local_shard == target)
          Runtime::trigger_event(done_event);
        else
          send_message();
      }
    }

    //--------------------------------------------------------------------------
    void GatherCollective::perform_collective_wait(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_shard == target); // should only be called on the target
#endif
      if (!done_event.has_triggered())
        done_event.lg_wait();
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
        if (local_shard == target)
          Runtime::trigger_event(done_event);
        else
          send_message();
      }
    }

    //--------------------------------------------------------------------------
    void GatherCollective::send_message(void)
    //--------------------------------------------------------------------------
    {
      // Convert to our local index
      const int local_index = convert_to_index(local_shard, target);
#ifdef DEBUG_LEGION
      assert(local_index >= shard_collective_radix);
#endif
      // Always round down to get our target index
      const int target_index = local_index / shard_collective_radix;
      // Then convert back to the target
      ShardID next = convert_to_shard(target_index, target);
      Serializer rez;
      {
        RezCheck z(rez);
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
      for (int idx = 0; idx < shard_collective_radix; idx++)
      {
        const int source_index = index * shard_collective_radix + idx;
        if (source_index > int(manager->total_shards))
          break;
        result++;
      }
      return result;
    }

    /////////////////////////////////////////////////////////////
    // All Gather Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AllGatherCollective::AllGatherCollective(ReplicateContext *ctx)
      : ShardCollective(ctx),       
        shard_collective_radix(ctx->get_shard_collective_radix()),
        shard_collective_log_radix(ctx->get_shard_collective_log_radix()),
        shard_collective_stages(ctx->get_shard_collective_stages()),
        shard_collective_participating_shards(
            ctx->get_shard_collective_participating_shards()),
        shard_collective_last_radix(ctx->get_shard_collective_last_radix()),
        shard_collective_last_log_radix(
            ctx->get_shard_collective_last_log_radix()),
        participating(local_shard < shard_collective_participating_shards) 
    //--------------------------------------------------------------------------
    { 
      if (participating)
      {
#ifdef DEBUG_LEGION
        assert(shard_collective_stages > 0);
#endif
        stage_notifications.resize(shard_collective_stages, 1);
        sent_stages.resize(shard_collective_stages, false);
        // Stage 0 always starts with 0 notifications since we'll 
        // explictcly arrive on it
	// Special case: if we expect a stage -1 message from a 
        // non-participating space, we'll count that as part of 
        // stage 0, it will make it a negative count, but the 
        // type is 'int' so we're good
        if ((shard_collective_stages > 0) &&
            (local_shard < 
             (manager->total_shards - shard_collective_participating_shards)))
          stage_notifications[0] = -1;
        else
          stage_notifications[0] = 0;
      }
      if (manager->total_shards > 1)
        done_event = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    AllGatherCollective::AllGatherCollective(ReplicateContext *ctx,
                                             CollectiveID id)
      : ShardCollective(ctx, id),
        shard_collective_radix(ctx->get_shard_collective_radix()),
        shard_collective_log_radix(ctx->get_shard_collective_log_radix()),
        shard_collective_stages(ctx->get_shard_collective_stages()),
        shard_collective_participating_shards(
            ctx->get_shard_collective_participating_shards()),
        shard_collective_last_radix(ctx->get_shard_collective_last_radix()),
        shard_collective_last_log_radix(
            ctx->get_shard_collective_last_log_radix()),
        participating(local_shard < shard_collective_participating_shards)
    //--------------------------------------------------------------------------
    {
      if (participating)
      {
#ifdef DEBUG_LEGION
        assert(shard_collective_stages > 0);
#endif
        stage_notifications.resize(shard_collective_stages, 1);
        sent_stages.resize(shard_collective_stages, false);
        // Stage 0 always starts with 0 notifications since we'll 
        // explictcly arrive on it
	// Special case: if we expect a stage -1 message from a 
        // non-participating space, we'll count that as part of 
        // stage 0, it will make it a negative count, but the 
        // type is 'int' so we're good
        if ((shard_collective_stages > 0) &&
            (local_shard < 
             (manager->total_shards - shard_collective_participating_shards)))
          stage_notifications[0] = -1;
        else
          stage_notifications[0] = 0;
      }
      if (manager->total_shards > 1)
        done_event = Runtime::create_rt_user_event();
    }

    //--------------------------------------------------------------------------
    AllGatherCollective::~AllGatherCollective(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void AllGatherCollective::perform_collective_sync(void)
    //--------------------------------------------------------------------------
    {
      perform_collective_async(); 
      perform_collective_wait();
    }

    //--------------------------------------------------------------------------
    void AllGatherCollective::perform_collective_async(void)
    //--------------------------------------------------------------------------
    {
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
          const bool all_stages_done = send_explicit_stage(0);
          if (all_stages_done)
            complete_exchange();
        }
      }
      else
      {
        // We are not a participating shard
        // so we just have to send a notification to one node
        send_explicit_stage(-1);
      }
    }

    //--------------------------------------------------------------------------
    void AllGatherCollective::perform_collective_wait(void)
    //--------------------------------------------------------------------------
    {
      if (manager->total_shards <= 1)
        return;
      if (!done_event.has_triggered())
        done_event.lg_wait();
    }

    //--------------------------------------------------------------------------
    void AllGatherCollective::handle_collective_message(Deserializer &derez)
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
          Runtime::trigger_event(done_event);
        else
          all_stages_done = send_explicit_stage(0); // we can now send stage 0
      }
      else
        all_stages_done = send_ready_stages();
      if (all_stages_done)
        complete_exchange();
    }

    //--------------------------------------------------------------------------
    bool AllGatherCollective::send_explicit_stage(int stage) 
    //--------------------------------------------------------------------------
    {
      bool all_stages_done = false;
      {
        AutoLock c_lock(collective_lock);
        // Mark that we're sending this stage
        if (stage >= 0)
        {
#ifdef DEBUG_LEGION
          assert(stage < int(stage_notifications.size()));
          if (stage == (shard_collective_stages-1))
            assert(stage_notifications[stage] < shard_collective_last_radix);
          else
            assert(stage_notifications[stage] < shard_collective_radix);
#endif
          stage_notifications[stage]++;
          sent_stages[stage] = true;
          // Check to see if all the stages are done
          all_stages_done = 
            (stage_notifications.back() == shard_collective_last_radix);
          if (all_stages_done)
          {
            for (int stage = 1; stage < shard_collective_stages; stage++)
            {
              if (stage_notifications[stage-1] == shard_collective_radix)
                continue;
              all_stages_done = false;
              break;
            }
          }
        }
      }
      if (stage == -1)
      {
        if (participating)
        {
          // Send back to the nodes that are not participating
          ShardID target = local_shard + shard_collective_participating_shards;
#ifdef DEBUG_LEGION
          assert(target < manager->total_shards);
#endif
          Serializer rez;
          construct_message(target, stage, rez);
          manager->send_collective_message(target, rez);
        }
        else
        {
          // Send to a node that is participating
          ShardID target = local_shard % shard_collective_participating_shards;
          Serializer rez;
          construct_message(target, stage, rez);
          manager->send_collective_message(target, rez);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(stage >= 0);
#endif
        if (stage == (shard_collective_stages-1))
        {
          for (int r = 1; r < shard_collective_last_radix; r++)
          {
            ShardID target = local_shard ^ 
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
            ShardID target = local_shard ^ 
              (r << (stage * shard_collective_log_radix));
#ifdef DEBUG_LEGION
            assert(int(target) < shard_collective_participating_shards);
#endif
            Serializer rez;
            construct_message(target, stage, rez);
            manager->send_collective_message(target, rez);
          }
        }
      }
      return all_stages_done;
    }

    //--------------------------------------------------------------------------
    bool AllGatherCollective::send_ready_stages(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(participating);
#endif
      // Iterate through the stages and send any that are ready
      // Remember that stages have to be done in order
      for (int stage = 1; stage < shard_collective_stages; stage++)
      {
        {
          AutoLock c_lock(collective_lock);
          // If this stage has already been sent then we can keep going
          if (sent_stages[stage])
            continue;
          // Check to see if we're sending this stage
          // We need all the notifications from the previous stage before
          // we can send this stage
          if (stage_notifications[stage-1] < shard_collective_radix)
            return false;
          // If we get here then we can send the stage
          sent_stages[stage] = true;
        }
        // Now we can do the send
        if (stage == (shard_collective_stages-1))
        {
          for (int r = 1; r < shard_collective_last_radix; r++)
          {
            ShardID target = local_shard ^ 
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
            ShardID target = local_shard ^ 
              (r << (stage * shard_collective_log_radix));
#ifdef DEBUG_LEGION
            assert(int(target) < shard_collective_participating_shards);
#endif
            Serializer rez;
            construct_message(target, stage, rez);
            manager->send_collective_message(target, rez);
          }
        }
      }
      // If we make it here, then we sent the last stage, check to see
      // if we've seen all the notifications for it
      AutoLock c_lock(collective_lock,1,false/*exclusive*/);
      return (stage_notifications.back() == shard_collective_last_radix);
    }

    //--------------------------------------------------------------------------
    void AllGatherCollective::construct_message(ShardID target, int stage,
                                                Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(manager->repl_id);
      rez.serialize(target);
      rez.serialize(collective_index);
      rez.serialize(stage);
      AutoLock c_lock(collective_lock, 1, false/*exclusive*/);
      pack_collective_stage(rez, stage);
    }

    //--------------------------------------------------------------------------
    void AllGatherCollective::unpack_stage(int stage, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(collective_lock);
      unpack_collective_stage(derez, stage);
    }

    //--------------------------------------------------------------------------
    void AllGatherCollective::complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      // We are done
      Runtime::trigger_event(done_event);
      // See if we have to send a message back to a
      // non-participating shard 
      if ((int(manager->total_shards) > shard_collective_participating_shards)
          && (int(local_shard) < int(manager->total_shards - 
                                      shard_collective_participating_shards)))
        send_explicit_stage(-1);
    }

    /////////////////////////////////////////////////////////////
    // Barrier Exchange Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    BarrierExchangeCollective::BarrierExchangeCollective(ReplicateContext *ctx,
                                  size_t win_size, std::vector<RtBarrier> &bars)
      : AllGatherCollective(ctx), window_size(win_size), barriers(bars)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    BarrierExchangeCollective::BarrierExchangeCollective(
                                           const BarrierExchangeCollective &rhs)
      : AllGatherCollective(rhs), window_size(0), barriers(rhs.barriers)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    BarrierExchangeCollective::~BarrierExchangeCollective(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    BarrierExchangeCollective& BarrierExchangeCollective::operator=(
                                           const BarrierExchangeCollective &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void BarrierExchangeCollective::exchange_barriers_async(void)
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
              RtBarrier(Realm::Barrier::create_barrier(manager->total_shards));
        }
      }
      // Now we can start the exchange from this shard 
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void BarrierExchangeCollective::wait_for_barrier_exchange(void)
    //--------------------------------------------------------------------------
    {
      // Wait for everything to be done
      perform_collective_wait();
#ifdef DEBUG_LEGION
      assert(local_barriers.size() == window_size);
#endif
      // Fill in the barrier vector with the barriers we've got from everyone
      barriers.resize(window_size);
      for (std::map<unsigned,RtBarrier>::const_iterator it = 
            local_barriers.begin(); it != local_barriers.end(); it++)
      {
#ifdef DEBUG_LEGION
        assert(it->first < window_size);
#endif
        barriers[it->first] = it->second;
      }
    }

    //--------------------------------------------------------------------------
    void BarrierExchangeCollective::pack_collective_stage(Serializer &rez, 
                                                          int stage) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(window_size);
      rez.serialize<size_t>(local_barriers.size());
      for (std::map<unsigned,RtBarrier>::const_iterator it = 
            local_barriers.begin(); it != local_barriers.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize(it->second);
      }
    }

    //--------------------------------------------------------------------------
    void BarrierExchangeCollective::unpack_collective_stage(Deserializer &derez,
                                                            int stage)
    //--------------------------------------------------------------------------
    {
      size_t other_window_size;
      derez.deserialize(other_window_size);
      if (other_window_size != window_size)
      {
        log_run.error("ERROR: Context configurations for control replicated "
                      "task %s were assigned different maximum window sizes "
                      "of %ld and %ld by the mapper which is illegal.",
                      context->owner_task->get_task_name(), window_size,
                      other_window_size);
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_MAPPER_OUTPUT);
      }
      size_t num_bars;
      derez.deserialize(num_bars);
      for (unsigned idx = 0; idx < num_bars; idx++)
      {
        unsigned index;
        derez.deserialize(index);
        derez.deserialize(local_barriers[index]);
      }
    }

    /////////////////////////////////////////////////////////////
    // Cross Product Collective 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    CrossProductCollective::CrossProductCollective(ReplicateContext *ctx)
      : AllGatherCollective(ctx)
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
                                                       int stage) const
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
                                                       ShardID target)
      : GatherCollective(ctx, target)
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
    bool ShardingGatherCollective::validate(ShardingID value) const
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
    // Field Descriptor Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FieldDescriptorExchange::FieldDescriptorExchange(ReplicateContext *ctx)
      : AllGatherCollective(ctx)
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
      }
      perform_collective_sync();
      return Runtime::merge_events(ready_events);
    }

    //--------------------------------------------------------------------------
    void FieldDescriptorExchange::pack_collective_stage(Serializer &rez,
                                                        int stage) const
    //--------------------------------------------------------------------------
    {
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
                                                 ShardID target)
      : GatherCollective(ctx, target)
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
      {
        AutoLock c_lock(collective_lock);
        ready_events.insert(ready_event);
        descriptors.insert(descriptors.end(), descs.begin(), descs.end());
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    const std::vector<FieldDataDescriptor>& 
                     FieldDescriptorGather::get_full_descriptors(ApEvent &ready)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      ready = Runtime::merge_events(ready_events);
      return descriptors;
    }

    /////////////////////////////////////////////////////////////
    // Future Broadcast 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureBroadcast::FutureBroadcast(ReplicateContext *ctx, CollectiveID id,
                                     ShardID source)
      : BroadcastCollective(ctx, id, source), result(NULL), result_size(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureBroadcast::FutureBroadcast(const FutureBroadcast &rhs)
      : BroadcastCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FutureBroadcast::~FutureBroadcast(void)
    //--------------------------------------------------------------------------
    {
      if (result != NULL)
        free(result);
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
      rez.serialize(result_size);
      if (result_size > 0)
        rez.serialize(result, result_size);
    }

    //--------------------------------------------------------------------------
    void FutureBroadcast::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(result_size);
      if (result_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(result == NULL);
#endif
        result = malloc(result_size);
        derez.deserialize(result, result_size);
      }
    }

    //--------------------------------------------------------------------------
    void FutureBroadcast::broadcast_future(const void *res, size_t size)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(result == NULL); 
#endif
      result_size = size;
      if (result_size > 0)
      {
        result = malloc(result_size);
        memcpy(result, res, result_size);
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void FutureBroadcast::receive_future(FutureImpl *f)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      if (result != NULL)
      {
        f->set_result(result, result_size, true/*own*/);
        result = NULL;
      }
      else
        f->set_result(NULL, 0, false/*own*/);
    }

    /////////////////////////////////////////////////////////////
    // Future Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureExchange::FutureExchange(ReplicateContext *ctx, size_t size)
      : AllGatherCollective(ctx), future_size(size)
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
    void FutureExchange::pack_collective_stage(Serializer &rez, int stage) const
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
#ifdef DEBUG_LEGION
        assert(results.find(shard) == results.end());
#endif
        void *buffer = malloc(future_size);
        derez.deserialize(buffer, future_size);
        results[shard] = buffer;
      }
    }

    //--------------------------------------------------------------------------
    void FutureExchange::reduce_futures(void *value, ReplIndexTask *target)
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
#ifdef DEBUG_LEGION
        assert(results.find(local_shard) == results.end());
#endif
        results[local_shard] = value;
      }
      perform_collective_sync();
      // Now we apply the shard results in order to ensure that we get
      // the same bitwise order across all the shards
      // No need for the lock anymore since we know we're done
      for (std::map<ShardID,void*>::const_iterator it = results.begin();
            it != results.end(); it++)
        target->fold_reduction_future(it->second, future_size, 
                                      false/*owner*/, true/*exclusive*/);
    }

    /////////////////////////////////////////////////////////////
    // Future Name Exchange 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    FutureNameExchange::FutureNameExchange(ReplicateContext *ctx,
                                           CollectiveID id)
      : AllGatherCollective(ctx, id)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    FutureNameExchange::FutureNameExchange(const FutureNameExchange &rhs)
      : AllGatherCollective(rhs)
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
    void FutureNameExchange::pack_collective_stage(Serializer &rez, 
                                                   int stage) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(results.size());
      for (std::map<DomainPoint,Future>::const_iterator it = 
            results.begin(); it != results.end(); it++)
      {
        rez.serialize(it->first);
        if (it->second.impl != NULL)
          rez.serialize(it->second.impl->did);
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
      for (unsigned idx = 0; idx < num_futures; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        DistributedID did;
        derez.deserialize(did);
        if (did > 0)
          results[point] = 
            Future(context->runtime->find_or_create_future(did, &mutator));
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
    MustEpochProcessorBroadcast::MustEpochProcessorBroadcast(
                                          ReplicateContext *ctx, ShardID origin)
      : BroadcastCollective(ctx, origin)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MustEpochProcessorBroadcast::MustEpochProcessorBroadcast(
                                         const MustEpochProcessorBroadcast &rhs)
      : BroadcastCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    MustEpochProcessorBroadcast::~MustEpochProcessorBroadcast(void)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    MustEpochProcessorBroadcast& MustEpochProcessorBroadcast::operator=(
                                         const MustEpochProcessorBroadcast &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void MustEpochProcessorBroadcast::pack_collective(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(origin_processors.size());
      for (unsigned idx = 0; idx < origin_processors.size(); idx++)
        rez.serialize(origin_processors[idx]);
    }

    //--------------------------------------------------------------------------
    void MustEpochProcessorBroadcast::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t num_procs;
      derez.deserialize(num_procs);
      origin_processors.resize(num_procs);
      for (unsigned idx = 0; idx < num_procs; idx++)
        derez.deserialize(origin_processors[idx]);
    }

    //--------------------------------------------------------------------------
    void MustEpochProcessorBroadcast::broadcast_processors(
                                       const std::vector<Processor> &processors)
    //--------------------------------------------------------------------------
    {
      origin_processors = processors;
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    bool MustEpochProcessorBroadcast::validate_processors(
                                       const std::vector<Processor> &processors)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
#ifdef DEBUG_LEGION
      assert(origin_processors.size() == processors.size());
#endif
      for (unsigned idx = 0; idx < processors.size(); idx++)
        if (processors[idx] != origin_processors[idx])
          return false;
      return true;
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Mapping Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochMappingExchange::MustEpochMappingExchange(ReplicateContext *ctx)
      : AllGatherCollective(ctx)
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
                                                         int stage) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(instances.size());
      for (std::map<unsigned,std::vector<DistributedID> >::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        rez.serialize(it->first);
        rez.serialize<size_t>(it->second.size());
        for (unsigned idx = 0; idx < it->second.size(); idx++)
          rez.serialize(it->second[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingExchange::unpack_collective_stage(Deserializer &derez,
                                                           int stage)
    //--------------------------------------------------------------------------
    {
      Runtime *runtime = manager->runtime;
      size_t num_mappings;
      derez.deserialize(num_mappings);
      for (unsigned idx1 = 0; idx1 < num_mappings; idx1++)
      {
        unsigned constraint_index;
        derez.deserialize(constraint_index);
#ifdef DEBUG_LEGION
        assert(constraint_index < results.size());
#endif
        std::vector<DistributedID> &dids = instances[constraint_index];
        std::vector<Mapping::PhysicalInstance> &mapping = 
          results[constraint_index];
        size_t num_instances;
        derez.deserialize(num_instances);
        dids.resize(num_instances);
        mapping.resize(num_instances);
        for (unsigned idx2 = 0; idx2 < num_instances; idx2++)
        {
          derez.deserialize(dids[idx2]);
          RtEvent ready;
          mapping[idx2].impl = 
            runtime->find_or_request_physical_manager(dids[idx2], ready);
          if (!ready.has_triggered())
            ready_events.insert(ready);
        }
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingExchange::exchange_must_epoch_mappings(
                ShardID shard_id, size_t total_shards, size_t total_constraints,
                std::vector<std::vector<Mapping::PhysicalInstance> > &mappings)
    //--------------------------------------------------------------------------
    {
      results.resize(total_constraints);
      {
        AutoLock c_lock(collective_lock);
        unsigned constraint_index = shard_id;
        for (unsigned idx1 = 0; idx1 < mappings.size(); 
              idx1++, constraint_index+=total_shards)
        {
#ifdef DEBUG_LEGION
          assert(constraint_index < total_constraints);
#endif
          results[constraint_index] = mappings[idx1];
          std::vector<DistributedID> &dids = instances[constraint_index];
          for (unsigned idx2 = 0; idx2 < mappings[idx1].size(); idx2++)
            dids[idx2] = mappings[idx1][idx2].impl->did;
        }
      }
      perform_collective_sync();
      // Wait for all the instances to be ready
      if (!ready_events.empty())
      {
        RtEvent ready = Runtime::merge_events(ready_events);
        if (!ready.has_triggered())
          ready.lg_wait();
      }
      mappings = results;
    }

    /////////////////////////////////////////////////////////////
    // Versioning Info Broadcast 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VersioningInfoBroadcast::VersioningInfoBroadcast(ReplicateContext *ctx,
                                                   CollectiveID id, ShardID own)
      : BroadcastCollective(ctx, id, own)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersioningInfoBroadcast::VersioningInfoBroadcast(
                                             const VersioningInfoBroadcast &rhs)
      : BroadcastCollective(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    VersioningInfoBroadcast::~VersioningInfoBroadcast(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    VersioningInfoBroadcast& VersioningInfoBroadcast::operator=(
                                             const VersioningInfoBroadcast &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void VersioningInfoBroadcast::pack_collective(Serializer &rez) const
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(versions.size());
      for (std::map<unsigned,LegionMap<DistributedID,FieldMask>::aligned>::
            const_iterator vit = versions.begin(); vit != versions.end(); vit++)
      {
        rez.serialize(vit->first);
        rez.serialize<size_t>(vit->second.size());
        for (LegionMap<DistributedID,FieldMask>::aligned::const_iterator it = 
              vit->second.begin(); it != vit->second.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersioningInfoBroadcast::unpack_collective(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(versions.empty());
#endif
      size_t num_versions;
      derez.deserialize(num_versions);
      for (unsigned idx1 = 0; idx1 < num_versions; idx1++)
      {
        unsigned index;
        derez.deserialize(index);
        LegionMap<DistributedID,FieldMask>::aligned &target = versions[index];
        size_t num_states;
        derez.deserialize(num_states);
        for (unsigned idx2 = 0; idx2 < num_states; idx2++)
        {
          DistributedID did;
          derez.deserialize(did);
          derez.deserialize(target[did]);
        }
      }
    }

    //--------------------------------------------------------------------------
    void VersioningInfoBroadcast::pack_advance_states(unsigned index,
                                                const VersionInfo &version_info)
    //--------------------------------------------------------------------------
    {
      version_info.capture_base_advance_states(versions[index]);
    }

    //--------------------------------------------------------------------------
    void VersioningInfoBroadcast::wait_for_states(
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait(); 
      std::set<RtEvent> wait_on;
      Runtime *runtime = context->runtime;
      // Now convert everything over to the results
      for (std::map<unsigned,LegionMap<DistributedID,FieldMask>::aligned>::
            const_iterator vit = versions.begin(); vit != versions.end(); vit++)
      {
        VersioningSet<> &target = results[vit->first];
        for (LegionMap<DistributedID,FieldMask>::aligned::const_iterator it = 
              vit->second.begin(); it != vit->second.end(); it++)
        {
          RtEvent ready;
          VersionState *state = 
            runtime->find_or_request_version_state(it->first, ready);
          ready = target.insert(state, it->second, runtime, ready);
          if (ready.exists() && !ready.has_triggered())
            wait_on.insert(ready);
        }
      }
      if (!wait_on.empty())
      {
        RtEvent wait_for = Runtime::merge_events(wait_on);
        wait_for.lg_wait();
      }
    }

    //--------------------------------------------------------------------------
    const VersioningSet<>& 
              VersioningInfoBroadcast::find_advance_states(unsigned index) const
    //--------------------------------------------------------------------------
    {
      std::map<unsigned,VersioningSet<> >::const_iterator finder = 
        results.find(index);
#ifdef DEBUG_LEGION
      assert(finder != results.end());
#endif
      return finder->second;
    }

  }; // namespace Internal
}; // namespace Legion

