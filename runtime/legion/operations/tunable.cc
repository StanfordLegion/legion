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

#include "legion/operations/tunable.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/utilities/collectives.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Tunable Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TunableOp::TunableOp(void) : Operation()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    TunableOp::~TunableOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Future TunableOp::initialize(
        InnerContext* ctx, const TunableLauncher& launcher,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      tunable_id = launcher.tunable;
      mapper_id = launcher.mapper;
      tag = launcher.tag;
      futures = launcher.futures;
      argsize = launcher.arg.get_size();
      if (argsize > 0)
      {
        arg = malloc(argsize);
        memcpy(arg, launcher.arg.get_ptr(), argsize);
      }
      return_type_size = launcher.return_type_size;
      result = Future(new FutureImpl(
          parent_ctx, true /*register*/,
          runtime->get_available_distributed_id(), get_provenance(), this));
      LegionSpy::log_tunable_operation(ctx->get_unique_id(), unique_op_id);
      const DomainPoint empty_point;
      LegionSpy::log_future_creation(
          unique_op_id, result.impl->did, empty_point);
      tunable_index = parent_ctx->get_tunable_index();
      return result;
    }

    //--------------------------------------------------------------------------
    void TunableOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      tunable_id = 0;
      mapper_id = 0;
      tag = 0;
      arg = nullptr;
      argsize = 0;
      tunable_index = 0;
      return_type_size = 0;
      instance = nullptr;
    }

    //--------------------------------------------------------------------------
    void TunableOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (arg != nullptr)
        free(arg);
      result = Future();
      futures.clear();
      if (instance != nullptr)
        delete instance;
      Operation::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* TunableOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TUNABLE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind TunableOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TUNABLE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void TunableOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      for (const Future& future : futures)
        future.impl->register_dependence(this);
    }

    //--------------------------------------------------------------------------
    void TunableOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> mapped_events, ready_events;
      for (const Future& future : futures)
      {
        future.impl->request_runtime_instance(this);
        const RtEvent ready = future.impl->find_runtime_instance_ready();
        if (ready.exists())
          ready_events.emplace_back(ready);
      }
      if (!mapped_events.empty())
        futures_mapped = Runtime::merge_events(mapped_events);
      // Make the instance if we have an upper bound size
      // and we have futures we'll likely need to defer on
      if (return_type_size < SIZE_MAX)
      {
        MemoryManager* manager =
            runtime->find_memory_manager(runtime->runtime_system_memory);
        TaskTreeCoordinates coordinates;
        compute_task_tree_coordinates(coordinates);
        // Safe to block here indefinitely waiting for unbounded pools
        instance = manager->create_future_instance(
            unique_op_id, coordinates, return_type_size,
            nullptr /*safe_for_unbounded_pools*/);
        complete_mapping(futures_mapped);
      }
      // Also make sure we wait for any execution fences that we have
      if (execution_fence_event.exists())
        ready_events.emplace_back(
            Runtime::protect_event(execution_fence_event));
      if (!ready_events.empty())
      {
        RtEvent ready = Runtime::merge_events(ready_events);
        if (ready.exists())
        {
          parent_ctx->add_to_trigger_execution_queue(this, ready);
          return;
        }
      }
      trigger_execution();
    }

    //--------------------------------------------------------------------------
    void TunableOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      MapperManager* mapper = runtime->find_mapper(
          parent_ctx->get_executing_processor(), mapper_id);
      Mapper::SelectTunableInput input;
      Mapper::SelectTunableOutput output;
      input.tunable_id = tunable_id;
      input.mapping_tag = tag;
      input.futures = futures;
      input.args = arg;
      input.size = argsize;
      mapper->invoke_select_tunable_value(
          parent_ctx->get_owner_task(), input, output);
      process_result(mapper, output.value, output.size);
      LegionSpy::log_tunable_value(
          parent_ctx->get_unique_id(), tunable_index, output.value,
          output.size);
      if (instance != nullptr)
      {
        if (output.size > return_type_size)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << *mapper << " returned tunable value of size "
                << output.size << " for selection of tunable value "
                << tunable_id
                << " but the upper bound size set by the launcher was only "
                << return_type_size << ".";
          error.raise();
        }
        // Copy the result into the instance
        FutureInstance* local = new FutureInstance(
            output.value, output.size, true /*external*/,
            output.take_ownership);
        const ApEvent done =
            instance->copy_from(local, this, ApEvent::NO_AP_EVENT);
        if (done.exists())
          record_completion_effect(done);
        result.impl->set_result(done, instance);
        // Future takes ownership of instance, so save local to instance
        // so we can reclaim it when it is safe to do so
        instance = local;
      }
      else
      {
        // Set and complete the future
        result.impl->set_local(
            output.value, output.size, output.take_ownership);
        complete_mapping();
      }
      complete_execution();
    }

    /////////////////////////////////////////////////////////////
    // Repl Tunable Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplTunableOp::ReplTunableOp(void) : TunableOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplTunableOp::~ReplTunableOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplTunableOp::activate(void)
    //--------------------------------------------------------------------------
    {
      TunableOp::activate();
      value_broadcast = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplTunableOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (value_broadcast != nullptr)
      {
        delete value_broadcast;
        value_broadcast = nullptr;
      }
      TunableOp::deactivate(false /*freeop*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplTunableOp::initialize_replication(ReplicateContext* repl_ctx)
    //--------------------------------------------------------------------------
    {
      if (runtime->safe_mapper)
      {
        legion_assert(value_broadcast == nullptr);
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        // We'll always make node zero the owner shard here
        if (repl_ctx->owner_shard->shard_id > 0)
          value_broadcast = new BufferBroadcast(
              repl_ctx, 0 /*owner shard*/, COLLECTIVE_LOC_100);
        else
          value_broadcast = new BufferBroadcast(repl_ctx, COLLECTIVE_LOC_100);
      }
    }

    //--------------------------------------------------------------------------
    void ReplTunableOp::process_result(
        MapperManager* mapper, void* buffer, size_t size) const
    //--------------------------------------------------------------------------
    {
      if (runtime->safe_mapper)
      {
        legion_assert(value_broadcast != nullptr);
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        if (repl_ctx->owner_shard->shard_id != value_broadcast->origin)
        {
          size_t expected_size = 0;
          const void* expected_buffer =
              value_broadcast->get_buffer(expected_size);
          if ((expected_size != size) ||
              (memcmp(buffer, expected_buffer, size) != 0))
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Mapper " << mapper->get_mapper_name()
                  << " returned different values for selection of "
                  << "tunable value " << tunable_id << " in parent task "
                  << parent_ctx->get_task_name() << " (UID "
                  << parent_ctx->get_unique_id() << ").";
            error.raise();
          }
        }
        else
          value_broadcast->broadcast(buffer, size);
      }
    }

  }  // namespace Internal
}  // namespace Legion
