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

#include "legion_context.h"
#include "legion_replication.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    const IndexSpace IndexSpaceReduction::identity = IndexSpace::NO_SPACE;
    const IndexPartitionID IndexPartitionReduction::identity = 0;
    const LegionColor ColorReduction::identity = INVALID_COLOR;
    const FieldSpace FieldSpaceReduction::identity = FieldSpace::NO_SPACE;
    const FieldID FieldReduction::identity = 0;
    const RegionTreeID LogicalRegionReduction::identity = 0;
    const long long TimingReduction::identity = 0;
    const bool TrueReduction::identity = false;
    const bool FalseReduction::identity = true;
#ifdef DEBUG_LEGION
    const ShardingID ShardingReduction::identity = UINT_MAX;
#endif

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
      sharding_functor = UINT_MAX;
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Check to see if we picked the same shardingID as everyone else
      // In theory this has already triggered, but we might need to 
      // explicitly wait to get realm to admit that
      if (!replicate_mapped_barrier.has_triggered())
        replicate_mapped_barrier.wait();
      ShardingID actual;
#ifndef NDEBUG
      bool valid = 
#endif
        replicate_mapped_barrier.get_result(&actual, sizeof(actual));
      assert(valid);
      if (actual != sharding_functor)
      {
        if (mapper != NULL)
          mapper = runtime->find_mapper(current_proc, map_id);
        log_run.error("ERROR: Mapper %s chose different sharding functions %d "
                      "and %d for individual task %s (UID %lld) in %s "
                      "(UID %lld)", mapper->get_mapper_name(), sharding_functor,
                      actual, get_task_name(), get_unique_id(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
        assert(false);
      }
#endif
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
      // Now we can trigger our result barrier indicating when we are mapped
      // If we're in debug mode we also reduce our ShardingID so we can 
      // confirm that all the mappers picked the same one for this operation
#ifdef DEBUG_LEGION
      // Debug arrival so contribute the sharding ID
      Runtime::phase_barrier_arrive(replicate_mapped_barrier, 1/*count*/,
              get_mapped_event(), &sharding_functor, sizeof(sharding_functor));
#else
      // Normal arrival
      Runtime::phase_barrier_arrive(replicate_mapped_barrier, 1/*count*/,
                                    get_mapped_event());
#endif
      // Now we can do the normal prepipeline stage
      IndividualTask::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::trigger_ready(void)
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
      else // We own it, so it goes on the ready queue
        enqueue_ready_operation(); 
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
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Check to see if we picked the same shardingID as everyone else
      // In theory this has already triggered, but we might need to 
      // explicitly wait to get realm to admit that
      if (!replicate_mapped_barrier.has_triggered())
        replicate_mapped_barrier.wait();
      ShardingID actual;
#ifndef NDEBUG
      bool valid = 
#endif
        replicate_mapped_barrier.get_result(&actual, sizeof(actual));
      assert(valid);
      if (actual != sharding_functor)
      {
        if (mapper != NULL)
          mapper = runtime->find_mapper(current_proc, map_id);
        log_run.error("ERROR: Mapper %s chose different sharding functions %d "
                      "and %d for index task %s (UID %lld) in %s (UID %lld)", 
                      mapper->get_mapper_name(), sharding_functor, 
                      actual, get_task_name(), get_unique_id(),
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
        assert(false);
      }
#endif 
      deactivate_index_task();
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
      // Get the sharding function implementation to use from our context
      ShardingFunction *function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
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
      else // We have valid points, so it goes on the ready queue
        enqueue_ready_operation();
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
      mapper = NULL;
    }

    //--------------------------------------------------------------------------
    void ReplIndexFillOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Check to see if we picked the same shardingID as everyone else
      // In theory this has already triggered, but we might need to 
      // explicitly wait to get realm to admit that
      if (!replicate_mapped_barrier.has_triggered())
        replicate_mapped_barrier.wait();
      ShardingID actual;
#ifndef NDEBUG
      bool valid = 
#endif
        replicate_mapped_barrier.get_result(&actual, sizeof(actual));
      assert(valid);
      if (actual != sharding_functor)
      {
        if (mapper != NULL)
          mapper = runtime->find_mapper(
              parent_ctx->get_executing_processor(), map_id);
        log_run.error("ERROR: Mapper %s chose different sharding functions %d "
                      "and %d for index fill in task %s (UID %lld)", 
                      mapper->get_mapper_name(), sharding_functor, actual,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
        assert(false);
      }
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
      // Now we can do the normal prepipeline stage
      IndexFillOp::trigger_prepipeline_stage();
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
      // Get the sharding function implementation to use from our context
      ShardingFunction *function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
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
      else // We have valid points, so it goes on the ready queue
        IndexFillOp::trigger_ready();
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
    void ReplCopyOp::activate(void)
    //--------------------------------------------------------------------------
    {
      activate_copy();
      sharding_functor = UINT_MAX;
    }

    //--------------------------------------------------------------------------
    void ReplCopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Check to see if we picked the same shardingID as everyone else
      // In theory this has already triggered, but we might need to 
      // explicitly wait to get realm to admit that
      if (!replicate_mapped_barrier.has_triggered())
        replicate_mapped_barrier.wait();
      ShardingID actual;
#ifndef NDEBUG
      bool valid = 
#endif
        replicate_mapped_barrier.get_result(&actual, sizeof(actual));
      assert(valid);
      if (actual != sharding_functor)
      {
        if (mapper != NULL)
          mapper = runtime->find_mapper(
              parent_ctx->get_executing_processor(), map_id);
        log_run.error("ERROR: Mapper %s chose different sharding functions %d "
                      "and %d for copy in task %s (UID %lld)", 
                      mapper->get_mapper_name(), sharding_functor, actual,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
        assert(false);
      }
#endif
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
      // Get the sharding function implementation to use from our context
      ShardingFunction *function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      // Figure out whether this shard owns this point
      ShardID owner_shard = function->find_owner(index_point, index_domain); 
      // If we own it we go on the queue, otherwise we complete early
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        // We don't own it, so we can pretend like we
        // mapped and executed this copy already
        complete_mapping();
        complete_execution();
      }
      else // We own it, so do the base call
        CopyOp::trigger_ready();
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
    }

    //--------------------------------------------------------------------------
    void ReplIndexCopyOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      // Check to see if we picked the same shardingID as everyone else
      // In theory this has already triggered, but we might need to 
      // explicitly wait to get realm to admit that
      if (!replicate_mapped_barrier.has_triggered())
        replicate_mapped_barrier.wait();
      ShardingID actual;
#ifndef NDEBUG
      bool valid = 
#endif
        replicate_mapped_barrier.get_result(&actual, sizeof(actual));
      assert(valid);
      if (actual != sharding_functor)
      {
        if (mapper != NULL)
          mapper = runtime->find_mapper(
              parent_ctx->get_executing_processor(), map_id);
        log_run.error("ERROR: Mapper %s chose different sharding functions %d "
                      "and %d for index copy in task %s (UID %lld)", 
                      mapper->get_mapper_name(), sharding_functor, actual,
                      parent_ctx->get_task_name(), parent_ctx->get_unique_id());
        assert(false);
      }
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
      // Now we can do the normal prepipeline stage
      IndexCopyOp::trigger_prepipeline_stage();
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
      // Get the sharding function implementation to use from our context
      ShardingFunction *function = 
        repl_ctx->shard_manager->find_sharding_function(sharding_functor);
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
      else // If we have any valid points do the base call
        IndexCopyOp::trigger_ready();
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
    }

    //--------------------------------------------------------------------------
    void ReplTimingOp::deactivate(void)
    //--------------------------------------------------------------------------
    {
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
        runtime->issue_runtime_meta_task(args, LG_LATENCY_PRIORITY, 
                                         this, timing_barrier);
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
        long long value;
#ifdef DEBUG_LEGION
#ifndef NDEBUG
        bool valid = 
#endif
#endif
          replicate_mapped_barrier.get_result(&value, sizeof(value));
#ifdef DEBUG_LEGION
        assert(valid);
#endif
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
              Runtime::phase_barrier_arrive(timing_barrier, 1/*count*/,
                  RtEvent::NO_RT_EVENT, &value, sizeof(value));
              break;
            }
          case MEASURE_MICRO_SECONDS:
            {
              long long value = Realm::Clock::current_time_in_microseconds();
              result.impl->set_result(&value, sizeof(value), false);
              Runtime::phase_barrier_arrive(timing_barrier, 1/*count*/,
                  RtEvent::NO_RT_EVENT, &value, sizeof(value));
              break;
            }
          case MEASURE_NANO_SECONDS:
            {
              long long value = Realm::Clock::current_time_in_nanoseconds();
              result.impl->set_result(&value, sizeof(value), false);
              Runtime::phase_barrier_arrive(timing_barrier, 1/*count*/,
                  RtEvent::NO_RT_EVENT, &value, sizeof(value));
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
        // We're the owner space so we have to make the allocation barriers
        //index_space_allocator_barrier(Realm::Barrier::create_barrier(
        index_space_allocator_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(1/*arrivers*/,
                REDOP_IS_REDUCTION, &IndexSpaceReduction::identity,
                sizeof(IndexSpaceReduction::identity)));
        index_partition_allocator_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(1/*arrivers*/,
                REDOP_IP_REDUCTION, &IndexPartitionReduction::identity,
                sizeof(IndexPartitionReduction::identity)));
        color_partition_allocator_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(1/*arrivers*/,
                REDOP_COLOR_REDUCTION, &ColorReduction::identity,
                sizeof(ColorReduction::identity)));
        field_space_allocator_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(1/*arrivers*/,
                REDOP_FS_REDUCTION, &FieldSpaceReduction::identity,
                sizeof(FieldSpaceReduction::identity)));
        field_allocator_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(1/*arrivers*/,
                REDOP_FID_REDUCTION, &FieldReduction::identity,
                sizeof(FieldReduction::identity)));
        logical_region_allocator_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(1/*arrivers*/,
                REDOP_LG_REDUCTION, &LogicalRegionReduction::identity,
                sizeof(LogicalRegionReduction::identity)));
        timing_measurement_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(1/*arrivers*/,
                REDOP_TIMING_REDUCTION, &TimingReduction::identity,
                sizeof(TimingReduction::identity)));
        disjointness_barrier = 
          RtBarrier(Realm::Barrier::create_barrier(1/*arrives*/,
                REDOP_TRUE_REDUCTION, &TrueReduction::identity,
                sizeof(TrueReduction::identity)));
        // Application barriers
        pending_partition_barrier = 
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
        // We're the owner so we have to destroy the allocation barriers
        index_space_allocator_barrier.destroy_barrier();
        index_partition_allocator_barrier.destroy_barrier();
        color_partition_allocator_barrier.destroy_barrier();
        field_space_allocator_barrier.destroy_barrier();
        field_allocator_barrier.destroy_barrier();
        logical_region_allocator_barrier.destroy_barrier();
        timing_measurement_barrier.destroy_barrier();
        disjointness_barrier.destroy_barrier();
        pending_partition_barrier.destroy_barrier();
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
      derez.deserialize(index_space_allocator_barrier);
      derez.deserialize(index_partition_allocator_barrier);
      derez.deserialize(color_partition_allocator_barrier);
      derez.deserialize(field_space_allocator_barrier);
      derez.deserialize(field_allocator_barrier);
      derez.deserialize(logical_region_allocator_barrier);
      derez.deserialize(timing_measurement_barrier);
      derez.deserialize(disjointness_barrier);
      derez.deserialize(pending_partition_barrier);
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
          rez.serialize(index_space_allocator_barrier);
          rez.serialize(index_partition_allocator_barrier);
          rez.serialize(color_partition_allocator_barrier);
          rez.serialize(field_space_allocator_barrier);
          rez.serialize(field_allocator_barrier);
          rez.serialize(logical_region_allocator_barrier);
          rez.serialize(timing_measurement_barrier);
          rez.serialize(disjointness_barrier);
          rez.serialize(pending_partition_barrier);
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
        runtime->send_control_rep_collective_stage(target_space, rez);
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
    /*static*/ void ShardManager::handle_collective_stage(Deserializer &derez,
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
      ShardingFunction *result = new ShardingFunction(functor, total_shards-1);
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
        done_event.wait();
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
    void GatherCollective::perform_collective_wait(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(local_shard == target); // should only be called on the target
#endif
      if (!done_event.has_triggered())
        done_event.wait();
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
        stage_notifications.resize(shard_collective_stages, 1);
        sent_stages.resize(shard_collective_stages, false);
        // Special case: if we expect a stage -1 message from a 
        // non-participating shard, we'll count that as part of stage 0
        if ((shard_collective_stages > 0) &&
            (local_shard < 
             (manager->total_shards - shard_collective_participating_shards)))
          stage_notifications[0]--;
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
          send_explicit_stage(0);
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
        done_event.wait();
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
      if (stage == -1)
      {
        if (!participating)
        {
          Runtime::trigger_event(done_event);
          return;
        }
        else
          send_explicit_stage(0); // we can now send stage 0
      }
      const bool all_stages_done = send_ready_stages();
      if (all_stages_done)
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
    }

    //--------------------------------------------------------------------------
    void AllGatherCollective::send_explicit_stage(int stage) 
    //--------------------------------------------------------------------------
    {
      {
        AutoLock c_lock(collective_lock);
        // Mark that we're sending this stage
        if (stage >= 0)
          sent_stages[stage] = true;
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
      for (int stage = 0; stage < shard_collective_stages; stage++)
      {
        {
          AutoLock c_lock(collective_lock);
          // If this stage has already been sent then we can keep going
          if (sent_stages[stage])
            continue;
          // Stage 0 should always be explicitly sent
          if (stage == 0)
            return false;
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
          // In debug mode we give a reduction to the application barriers
          // so that we can check that they all used the same ShardingID
          local_barriers[index] = 
            RtBarrier(Realm::Barrier::create_barrier(manager->total_shards,
                  REDOP_SID_REDUCTION, &ShardingReduction::identity,
                  sizeof(ShardingReduction::identity)));
#else
          local_barriers[index] = 
              RtBarrier(Realm::Barrier::create_barrier(manager->total_shards));
#endif
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

  }; // namespace Internal
}; // namespace Legion

