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

#include "legion/operations/allreduce.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/tracing/recognizer.h"
#include "legion/tracing/template.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // All Reduce Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    AllReduceOp::AllReduceOp(void) : MemoizableOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    AllReduceOp::~AllReduceOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    Future AllReduceOp::initialize(
        InnerContext* ctx, const FutureMap& fm, ReductionOpID redid,
        bool is_deterministic, MapperID map_id, MappingTagID t,
        Provenance* provenance, Future initial_value)
    //--------------------------------------------------------------------------
    {
      initialize_operation(ctx, provenance);
      future_map = fm;
      redop_id = redid;
      redop = runtime->get_reduction(redop_id);
      serdez_redop_fns = Runtime::get_serdez_redop_fns(redop_id);
      result = Future(new FutureImpl(
          parent_ctx, true /*register*/,
          runtime->get_available_distributed_id(), get_provenance(), this));
      if (serdez_redop_fns == nullptr)
        result.impl->set_future_result_size(
            redop->sizeof_rhs, runtime->address_space);
      this->initial_value = initial_value;

      mapper_id = map_id;
      tag = t;
      deterministic = is_deterministic;
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        LegionSpy::log_all_reduce_operation(ctx->get_unique_id(), unique_op_id);
        const DomainPoint empty_point;
        LegionSpy::log_future_creation(
            unique_op_id, result.impl->did, empty_point);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      MemoizableOp::activate();
      redop_id = 0;
      future_result_size = 0;
      serdez_redop_buffer = nullptr;
      serdez_upper_bound = SIZE_MAX;
      serdez_redop_instance = nullptr;
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      future_map = FutureMap();
      result = Future();
      initial_value = Future();
      sources.clear();
      targets.clear();
      target_memories.clear();
      map_applied_conditions.clear();
      if (serdez_redop_buffer != nullptr)
        free(serdez_redop_buffer);
      if (serdez_redop_instance != nullptr)
        delete serdez_redop_instance;
      MemoizableOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* AllReduceOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[ALL_REDUCE_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind AllReduceOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return ALL_REDUCE_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      if (initial_value.impl != nullptr)
        initial_value.impl->register_dependence(this);
      future_map.impl->register_dependence(this);
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::prepare_future(
        std::vector<RtEvent>& preconditions, FutureImpl* future)
    //--------------------------------------------------------------------------
    {
      future->request_runtime_instance(this);
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::populate_sources(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(sources.empty());
      future_map.impl->get_all_futures(sources);
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::fold_serdez(FutureImpl* impl)
    //--------------------------------------------------------------------------
    {
      if (impl == nullptr)
        return;
      size_t src_size = 0;
      const void* source = impl->find_runtime_buffer(parent_ctx, src_size);
      serdez_redop_fns->fold_fn(
          redop, serdez_redop_buffer, future_result_size, source);
      LegionSpy::log_future_use(unique_op_id, impl->did);
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::all_reduce_serdez(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(serdez_redop_fns != nullptr);
      // Initialize here so that we can set the initial value future
      future_result_size = 0;
      serdez_redop_fns->init_fn(redop, serdez_redop_buffer, future_result_size);
      fold_serdez(initial_value.impl);
      for (const std::pair<const DomainPoint, FutureImpl*>& src : sources)
        fold_serdez(src.second);
    }

    //--------------------------------------------------------------------------
    ApEvent AllReduceOp::finalize_serdez_targets(void)
    //--------------------------------------------------------------------------
    {
      // Now that we've got the output instances we copy the result to
      // each of the targets, we're done when the copies are done
      // create an external instance for the current allocation
      FutureInstance* serdez_redop_instance = new FutureInstance(
          serdez_redop_buffer, future_result_size, true /*external*/,
          false /*own allocation*/);
      std::vector<ApEvent> done_events;
      for (FutureInstance* target : targets)
      {
        ApEvent done = target->copy_from(
            serdez_redop_instance, this, ApEvent::NO_AP_EVENT);
        if (done.exists())
          done_events.emplace_back(done);
      }
      if (!done_events.empty())
        return Runtime::merge_events(nullptr, done_events);
      else
        return ApEvent::NO_AP_EVENT;
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::subscribe_to_future(
        std::vector<RtEvent>& ready_events, FutureImpl* future)
    //--------------------------------------------------------------------------
    {
      const RtEvent ready = future->find_runtime_instance_ready();
      if (ready.exists())
        ready_events.emplace_back(ready);
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      // Invoke the mapper to do figure out where to put the data
      invoke_mapper();
      // Then we can perform the all-reduce
      perform_allreduce();
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_replay_operation(unique_op_id);
      tpl->get_allreduce_mapping(this, target_memories, future_result_size);
      perform_allreduce();
    }

    //--------------------------------------------------------------------------
    bool AllReduceOp::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      Murmur3Hasher hasher;
      hasher.hash(get_operation_kind());
      return recorder.record_operation_hash(this, hasher, opidx);
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::perform_allreduce(void)
    //--------------------------------------------------------------------------
    {
      // Call from both trigger_mapping and trigger_replay
      // Request host buffers for any of the source instances
      populate_sources();
      // Always make sure we'll have buffers ready on the host for us to
      // access in order to use for doing the all-reduce
      for (const std::pair<const DomainPoint, FutureImpl*>& src : sources)
        prepare_future(map_applied_conditions, src.second);
      if (initial_value.impl != nullptr)
        prepare_future(map_applied_conditions, initial_value.impl);
      if (future_result_size < SIZE_MAX)
      {
        // We can only make the future results now if we have an actual
        // future result size to use which might not be the case if the
        // mapper didn't specify an upper bound on the size of the results
        create_future_instances();
        // We're done with our mapping at the point we've made all the instances
        if (!map_applied_conditions.empty())
          complete_mapping(Runtime::merge_events(map_applied_conditions));
        else
          complete_mapping();
      }
      // Subscribe to all the futures and then perform the computation
      std::vector<RtEvent> ready_events;
      for (const std::pair<const DomainPoint, FutureImpl*>& src : sources)
        subscribe_to_future(ready_events, src.second);
      if (initial_value.impl != nullptr)
        subscribe_to_future(ready_events, initial_value.impl);
      // Also make sure we wait for any execution fences that we have
      if ((serdez_redop_fns != nullptr) && execution_fence_event.exists())
        ready_events.emplace_back(
            Runtime::protect_event(execution_fence_event));
      if (!ready_events.empty())
      {
        const RtEvent ready = Runtime::merge_events(ready_events);
        if (ready.exists())
        {
          parent_ctx->add_to_trigger_execution_queue(this, ready);
          return;
        }
      }
      // If we get here then we can trigger execution immediately
      trigger_execution();
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::trigger_execution(void)
    //--------------------------------------------------------------------------
    {
      ApEvent done;
      RtEvent executed;
      if (serdez_redop_fns != nullptr)
      {
        all_reduce_serdez();
        if (serdez_upper_bound == SIZE_MAX)
        {
          // Make the instances for the target memories
          create_future_instances();
          // We're done with our mapping now that we've made all the instances
          if (!map_applied_conditions.empty())
            complete_mapping(Runtime::merge_events(map_applied_conditions));
          else
            complete_mapping();
        }
        // Check that the result is smaller than the bound
        if (serdez_upper_bound < future_result_size)
        {
          Processor exec_proc = parent_ctx->get_executing_processor();
          MapperManager* mapper = runtime->find_mapper(exec_proc, mapper_id);
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output. Mapper " << *mapper
                << " specified an upper bound of " << serdez_upper_bound
                << " bytes for future map all reduce with serdez redop "
                << redop_id
                << ". However, the actual size of the reduced value is "
                << future_result_size
                << " bytes which exceeds the specified upper bound.";
          error.raise();
        }
        done = finalize_serdez_targets();
      }
      else
        done = all_reduce_redop(executed);
      if (done.exists())
        record_completion_effect(done);
      result.impl->set_results(done, targets);
      complete_execution(executed);
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
      Mapper::FutureMapReductionInput input{tag};
      Mapper::FutureMapReductionOutput output;
      Processor exec_proc = parent_ctx->get_executing_processor();
      MapperManager* mapper = runtime->find_mapper(exec_proc, mapper_id);
      mapper->invoke_map_future_map_reduction(this, input, output);
      serdez_upper_bound = output.serdez_upper_bound;
      if (!output.destination_memories.empty())
      {
        if (output.destination_memories.size() > 1)
        {
          std::set<Memory> unique_memories;
          for (std::vector<Memory>::iterator it =
                   output.destination_memories.begin();
               it != output.destination_memories.end();
               /*nothing*/)
          {
            if (!it->exists())
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Invalid mapper output. Mapper " << *mapper
                    << " requested future map reduction future be mapped to a "
                    << "NO_MEMORY for " << *this
                    << " which is illegal. All requests for mapping output "
                       "futures "
                    << "must be mapped to actual memories.";
              error.raise();
            }
            if (unique_memories.find(*it) == unique_memories.end())
            {
              unique_memories.insert(*it);
              it++;
            }
            else
              it = output.destination_memories.erase(it);
          }
        }
        else if (!output.destination_memories.front().exists())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output. Mapper " << *mapper
                << " requested future map reduction future be mapped to a "
                << "NO_MEMORY for future map " << *this
                << " which is illegal. All requests for mapping output futures "
                << "must be mapped to actual memories.";
          error.raise();
        }
        target_memories.swap(output.destination_memories);
      }
      else
        target_memories.emplace_back(runtime->runtime_system_memory);
      // Compute the future reduction size
      if (serdez_redop_fns == nullptr)
        future_result_size = redop->sizeof_rhs;
      else
        future_result_size = serdez_upper_bound;
      if (is_recording())
      {
        if (future_result_size == SIZE_MAX)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output. Mapper " << *mapper
                << " did not specify an upper bound on serdez future " << *this
                << " being traced. All serdez future reductions being captured "
                << "in traces must provide an upper bound on the size of the "
                << "future result.";
          error.raise();
        }
        const TraceInfo trace_info(this);
        const TraceLocalID tlid = get_trace_local_id();
        trace_info.record_future_allreduce(
            tlid, target_memories, future_result_size);
      }
    }

    //--------------------------------------------------------------------------
    void AllReduceOp::create_future_instances(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(targets.empty());
      legion_assert(!target_memories.empty());
      targets.reserve(target_memories.size());
      // If we don't have serdez functions or the upper bound is not set
      // then we can use the future_result_size since we know it is the
      // right size for the futures, otherwise we need to trust the
      // serdez_upper_bound size as the size of these futures.
      const size_t result_size =
          ((serdez_redop_fns == nullptr) || (serdez_upper_bound == SIZE_MAX)) ?
              future_result_size :
              serdez_upper_bound;
      TaskTreeCoordinates coordinates;
      compute_task_tree_coordinates(coordinates);
      int runtime_visible = -1;
      for (const Memory& target : target_memories)
      {
        if ((runtime_visible < 0) && FutureInstance::check_meta_visible(target))
          runtime_visible = targets.size();
        MemoryManager* manager = runtime->find_memory_manager(target);
        // Safe to block here indefinitely waiting for unbounded pools
        FutureInstance* instance = manager->create_future_instance(
            unique_op_id, coordinates, result_size,
            nullptr /*safe for unbounded pools*/);
        targets.emplace_back(instance);
      }
      // This is an important optimization: if we're doing a small
      // reduction value we always want the reduction instance to
      // be somewhere meta visible for performance reasons, so we
      // make a meta-visible instance if we don't have one
      if ((runtime_visible < 0) && (serdez_redop_fns == nullptr) &&
          (redop->sizeof_rhs <= LEGION_MAX_RETURN_SIZE))
      {
        runtime_visible = targets.size();
        targets.emplace_back(FutureInstance::create_local(
            &redop->identity, redop->sizeof_rhs, false /*own*/));
      }
      if (runtime_visible > 0)
        std::swap(targets.front(), targets[runtime_visible]);
    }

    //--------------------------------------------------------------------------
    ApEvent AllReduceOp::init_redop_target(FutureInstance* target)
    //--------------------------------------------------------------------------
    {
      if (parent_ctx->get_task()->get_shard_id() == 0)
      {
        FutureImpl* init = initial_value.impl;
        if (init != nullptr)
          return init->copy_to(target, this, execution_fence_event);
      }
      return target->initialize(redop, this, execution_fence_event);
    }

    //--------------------------------------------------------------------------
    ApEvent AllReduceOp::all_reduce_redop(RtEvent& executed)
    //--------------------------------------------------------------------------
    {
      std::vector<ApEvent> preconditions(targets.size());
      for (unsigned idx = 0; idx < targets.size(); idx++)
        preconditions[idx] = init_redop_target(targets[idx]);
      std::vector<ApEvent> postconditions;
      if (deterministic)
      {
        for (const std::pair<const DomainPoint, FutureImpl*>& src : sources)
        {
          for (unsigned idx = 0; idx < targets.size(); idx++)
            preconditions[idx] = src.second->reduce_to(
                targets[idx], this, redop_id, redop, true /*exclusive*/,
                preconditions[idx]);
          LegionSpy::log_future_use(unique_op_id, src.second->did);
        }
        for (const ApEvent& pre : preconditions)
          if (pre.exists())
            postconditions.emplace_back(pre);
      }
      else
      {
        for (const std::pair<const DomainPoint, FutureImpl*>& src : sources)
        {
          for (unsigned idx = 0; idx < targets.size(); idx++)
          {
            const ApEvent done = src.second->reduce_to(
                targets[idx], this, redop_id, redop, false /*exclusive*/,
                preconditions[idx]);
            if (done.exists())
              postconditions.emplace_back(done);
          }
          LegionSpy::log_future_use(unique_op_id, src.second->did);
        }
      }
      if (!postconditions.empty())
        return Runtime::merge_events(nullptr, postconditions);
      else
        return ApEvent::NO_AP_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Repl All Reduce Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplAllReduceOp::ReplAllReduceOp(void) : AllReduceOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplAllReduceOp::~ReplAllReduceOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      legion_assert(redop != nullptr);
      legion_assert(serdez_redop_collective == nullptr);
      legion_assert(all_reduce_collective == nullptr);
      legion_assert(reduction_collective == nullptr);
      legion_assert(broadcast_collective == nullptr);
      if (serdez_redop_fns == nullptr)
      {
        if (deterministic)
        {
          broadcast_collective = new FutureBroadcastCollective(
              ctx, COLLECTIVE_LOC_65, 0 /*origin shard*/, this);
          reduction_collective = new FutureReductionCollective(
              ctx, COLLECTIVE_LOC_66, 0 /*origin shard*/, this,
              broadcast_collective, redop, redop_id);
        }
        else
          all_reduce_collective = new FutureAllReduceCollective(
              this, COLLECTIVE_LOC_97, ctx, redop_id, redop);
      }
      else
        serdez_redop_collective = new BufferExchange(ctx, COLLECTIVE_LOC_97);
    }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::activate(void)
    //--------------------------------------------------------------------------
    {
      AllReduceOp::activate();
      serdez_redop_collective = nullptr;
      all_reduce_collective = nullptr;
      reduction_collective = nullptr;
      broadcast_collective = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (serdez_redop_collective != nullptr)
        delete serdez_redop_collective;
      if (all_reduce_collective != nullptr)
        delete all_reduce_collective;
      if (reduction_collective != nullptr)
        delete reduction_collective;
      if (broadcast_collective != nullptr)
        delete broadcast_collective;
      AllReduceOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::populate_sources(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(sources.empty());
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      future_map.impl->get_shard_local_futures(
          repl_ctx->owner_shard->shard_id, sources);
    }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::create_future_instances(void)
    //--------------------------------------------------------------------------
    {
      // Do the base call first
      AllReduceOp::create_future_instances();
      // Now check to see if we need to make a shadow instance for
      // the all-reduce future collective
      if (all_reduce_collective != nullptr)
      {
        legion_assert(!targets.empty());
        legion_assert(serdez_redop_fns == nullptr);
        FutureInstance* target = targets.front();
        // If the instance is in a memory we cannot see or is "too big"
        // then we need to make the shadow instance for the future
        // all-reduce collective to use now while still in the mapping stage
        if ((!target->is_meta_visible) ||
            (target->size > LEGION_MAX_RETURN_SIZE))
        {
          MemoryManager* manager = runtime->find_memory_manager(target->memory);
          TaskTreeCoordinates coordinates;
          compute_task_tree_coordinates(coordinates);
          // Safe to block here indefinitely waiting for unbounded pools
          FutureInstance* shadow_instance = manager->create_future_instance(
              unique_op_id, coordinates, redop->sizeof_rhs,
              nullptr /*safe_for_unbounded_pools*/);
          all_reduce_collective->set_shadow_instance(shadow_instance);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplAllReduceOp::all_reduce_serdez(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(serdez_redop_fns != nullptr);
      future_result_size = 0;
      serdez_redop_fns->init_fn(redop, serdez_redop_buffer, future_result_size);
      // Only include the initial value one time for control replication
      // to avoid double inclusion
      if (parent_ctx->get_task()->get_shard_id() == 0)
        fold_serdez(initial_value.impl);
      for (const std::pair<const DomainPoint, FutureImpl*>& src : sources)
        fold_serdez(src.second);
      // Now we need an all-to-all to get the values from other shards
      const std::map<ShardID, std::pair<void*, size_t> >& remote_buffers =
          serdez_redop_collective->exchange_buffers(
              serdez_redop_buffer, future_result_size, deterministic);
      if (deterministic)
      {
        // Reset this back to empty so we can reduce in order across shards
        // Note the serdez_redop_collective took ownership of deleting
        // the buffer in this case so we know that it is not leaking
        serdez_redop_buffer = nullptr;
        for (const std::pair<const ShardID, std::pair<void*, size_t> >& it :
             remote_buffers)
        {
          if (serdez_redop_buffer == nullptr)
          {
            future_result_size = it.second.second;
            serdez_redop_buffer = malloc(future_result_size);
            memcpy(serdez_redop_buffer, it.second.first, future_result_size);
          }
          else
            serdez_redop_fns->fold_fn(
                redop, serdez_redop_buffer, future_result_size,
                it.second.first);
        }
      }
      else
      {
        for (const std::pair<const ShardID, std::pair<void*, size_t> >& it :
             remote_buffers)
        {
          legion_assert(it.first != serdez_redop_collective->local_shard);
          serdez_redop_fns->fold_fn(
              redop, serdez_redop_buffer, future_result_size, it.second.first);
        }
      }
    }

    //--------------------------------------------------------------------------
    ApEvent ReplAllReduceOp::all_reduce_redop(RtEvent& executed)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const DomainPoint, FutureImpl*>& src : sources)
      {
        FutureImpl* impl = src.second;
        const size_t source_size = impl->get_untyped_size();
        if (source_size != redop->sizeof_rhs)
        {
          Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
          error
              << "Future in future map reduction in task "
              << parent_ctx->get_task_name() << " (UID "
              << parent_ctx->get_unique_id() << ") does not "
              << "have the right input size for the given reduction operator. "
              << "Future has size " << source_size
              << " bytes but reduction operator expects "
              << "RHS inputs of " << redop->sizeof_rhs << " bytes.";
          error.raise();
        }
        LegionSpy::log_future_use(unique_op_id, impl->did);
      }
      legion_assert(!targets.empty());
      // We're going to need to do an all-reduce between the shards so
      // we'll just do our local reductions into the first target initially
      // and then we'll broadcast the result to the targets afterwards
      FutureInstance* local_target = targets.front();
      ApEvent local_precondition = init_redop_target(local_target);
      if (deterministic)
      {
        for (const std::pair<const DomainPoint, FutureImpl*>& src : sources)
        {
          local_precondition = src.second->reduce_to(
              local_target, this, redop_id, redop, true /*exclusive*/,
              local_precondition);
          LegionSpy::log_future_use(unique_op_id, src.second->did);
        }
      }
      else
      {
        std::vector<ApEvent> postconditions;
        for (const std::pair<const DomainPoint, FutureImpl*>& src : sources)
        {
          const ApEvent postcondition = src.second->reduce_to(
              local_target, this, redop_id, redop, false /*exclusive*/,
              local_precondition);
          if (postcondition.exists())
            postconditions.emplace_back(postcondition);
          LegionSpy::log_future_use(unique_op_id, src.second->did);
        }
        if (!postconditions.empty())
          local_precondition = Runtime::merge_events(nullptr, postconditions);
      }
      if (all_reduce_collective == nullptr)
      {
        reduction_collective->async_reduce(targets.front(), local_precondition);
        local_precondition = broadcast_collective->finished;
        if (broadcast_collective->is_origin())
          executed = reduction_collective->get_done_event();
        else
          executed = broadcast_collective->async_broadcast(
              targets.front(), ApEvent::NO_AP_EVENT,
              reduction_collective->get_done_event());
      }
      else
        executed = all_reduce_collective->async_reduce(
            targets.front(), local_precondition);
      // Finally do the copy out to all the other targets
      if (targets.size() > 1)
      {
        std::vector<ApEvent> broadcast_events(targets.size());
        broadcast_events[0] = local_precondition;
        broadcast_events[1] =
            targets[1]->copy_from(local_target, this, broadcast_events[0]);
        bool need_merge = false;
        for (unsigned idx = 1; idx < targets.size(); idx++)
        {
          if (targets.size() <= (2 * idx))
            break;
          broadcast_events[2 * idx] = targets[2 * idx]->copy_from(
              targets[idx], this, broadcast_events[idx]);
          if (broadcast_events[2 * idx].exists())
            need_merge = true;
          if (targets.size() <= (2 * idx + 1))
            break;
          broadcast_events[2 * idx + 1] = targets[2 * idx + 1]->copy_from(
              targets[idx], this, broadcast_events[idx]);
          if (broadcast_events[2 * idx + 1].exists())
            need_merge = true;
        }
        if (need_merge)
          local_precondition = Runtime::merge_events(nullptr, broadcast_events);
      }
      return local_precondition;
    }

  }  // namespace Internal
}  // namespace Legion
