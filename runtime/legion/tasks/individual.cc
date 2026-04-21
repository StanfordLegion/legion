/* Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "legion/tasks/individual.h"
#include "legion/analysis/logical.h"
#include "legion/contexts/replicate.h"
#include "legion/api/functors_impl.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/processor.h"
#include "legion/managers/shard.h"
#include "legion/operations/mustepoch.h"
#include "legion/tracing/recognizer.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Individual Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndividualTask::IndividualTask(void) : SingleTask()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndividualTask::~IndividualTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndividualTask::activate(void)
    //--------------------------------------------------------------------------
    {
      SingleTask::activate();
      output_regions_registered = RtEvent::NO_RT_EVENT;
      concurrent_precondition = RtUserEvent::NO_RT_USER_EVENT;
      concurrent_postcondition = RtEvent::NO_RT_EVENT;
      orig_task = this;
      remote_unique_id = get_unique_id();
      sent_remotely = false;
      top_level_task = false;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      predicate_false_result.clear();
      // Remove our reference on the future
      result = Future();
      predicate_false_future = Future();
      output_region_options.clear();
      SingleTask::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    Future IndividualTask::initialize_task(
        InnerContext* ctx, const TaskLauncher& launcher, Provenance* provenance,
        bool top_level /*=false*/, bool must_epoch_launch /*=false*/,
        std::vector<OutputRequirement>* outputs /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      regions = launcher.region_requirements;
      futures = launcher.futures;
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.argument.get_size();
      if (arglen > 0)
      {
        arg_manager.save_buffer(launcher.argument.get_ptr(), arglen);
        args = arg_manager.get_buffer();
      }
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      index_point = launcher.point;
      index_domain = Domain(index_point, index_point);
      sharding_space = launcher.sharding_space;
      is_index_space = false;
      initialize_base_task(ctx, launcher.predicate, task_id, provenance);
      // If the task has any output requirements, we create fresh region names
      // return them back to the user
      if (outputs != nullptr)
      {
        create_output_regions(*outputs);
        if (launcher.predicate != Predicate::TRUE_PRED)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Output requirements are disallowed for tasks launched with "
                << "predicates, but predicated task launch for task " << *this
                << " in parent task " << *parent_ctx
                << " is used with output requirements.";
          error.raise();
        }
        if (get_trace() != nullptr)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Output requirements are disallowed for tasks launched "
                << "inside traces. Task " << *this << " in parent task "
                << *parent_ctx << " has output requirements in trace "
                << get_trace()->get_trace_id() << ".";
          error.raise();
        }
      }
      if (launcher.predicate != Predicate::TRUE_PRED &&
          !launcher.elide_future_return)
      {
        if (launcher.predicate_false_future.impl != nullptr)
          predicate_false_future = launcher.predicate_false_future;
        else if (launcher.predicate_false_result.get_size() > 0)
          predicate_false_result.save_buffer(
              launcher.predicate_false_result.get_ptr(),
              launcher.predicate_false_result.get_size());
      }
      if (launcher.local_function_task)
      {
        if (!regions.empty())
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Local function task launch for task " << get_task_name()
                << " in parent task " << parent_ctx->get_task_name() << " (UID "
                << parent_ctx->get_unique_id() << ") has " << regions.size()
                << " region requirements. Local function tasks "
                << "are not permitted to have any region requirements.";
          error.raise();
        }
        local_function = true;
      }
      // Get a future from the parent context to use as the result
      if (launcher.elide_future_return)
        elide_future_return = true;
      else
      {
        future_return_size = launcher.future_return_size;
        if (!must_epoch_launch)
          result = create_future();
      }
      // Make sure you do this after making the output regions
      compute_parent_indexes(false /*force*/);
      // If this is the top-level task we can record some extra properties
      if (top_level)
        this->top_level_task = true;
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        if (top_level)
          LegionSpy::log_top_level_task(
              task_id, parent_ctx->get_unique_id(), unique_op_id,
              get_task_name());
        // Tracking as long as we are not part of a must epoch operation
        if (!must_epoch_launch || top_level)
          LegionSpy::log_individual_task(
              parent_ctx->get_unique_id(), unique_op_id, task_id,
              get_task_name());
        for (const PhaseBarrier& barrier : launcher.wait_barriers)
        {
          ApEvent e = Runtime::get_previous_phase(barrier.phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    Future IndividualTask::create_future(void)
    //--------------------------------------------------------------------------
    {
      FutureImpl* impl = new FutureImpl(
          parent_ctx, true /*register*/,
          runtime->get_available_distributed_id(), get_provenance(), this);
      LegionSpy::log_future_creation(unique_op_id, impl->did, index_point);
      return Future(impl);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::prepare_map_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch != nullptr);
      set_origin_mapped(true);
      if (!elide_future_return)
      {
        FutureMap map = must_epoch->get_future_map();
        result = map.impl->get_future(index_point, true /*internal only*/);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::create_output_regions(
        std::vector<OutputRequirement>& outputs)
    //--------------------------------------------------------------------------
    {
      output_region_options.resize(outputs.size());
      Provenance* provenance = get_provenance();
      for (unsigned idx = 0; idx < outputs.size(); idx++)
      {
        OutputRequirement& req = outputs[idx];
        output_region_options[idx] =
            OutputOptions(false, req.valid_requirement, false /*grouped*/);

        if (!req.valid_requirement)
        {
          // Create a deferred index space
          IndexSpace index_space =
              parent_ctx->create_unbound_index_space(req.type_tag, provenance);
          // Create an output region
          LogicalRegion region = parent_ctx->create_logical_region(
              index_space, req.field_space, false /*local region*/, provenance,
              true /*output region*/);

          // Set the region back to the output requirement so the caller
          // can use it for downstream tasks
          req.region = region;
          req.parent = region;
          req.flags |= LEGION_CREATED_OUTPUT_REQUIREMENT_FLAG;
        }
        req.privilege = LEGION_WRITE_DISCARD;

        // Store the output requirement in the task
        output_regions.emplace_back(req);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      if (!options_selected)
      {
        const bool inline_task = select_task_options(false /*prioritize*/);
        if (inline_task)
        {
          Warning warning;
          warning << "Mapper " << mapper->get_mapper_name()
                  << " requested to inline task " << get_task_name() << " (UID "
                  << get_unique_id()
                  << ") but the 'enable_inlining' option was "
                  << "not set on the task launcher so the request is "
                  << "being ignored.";
          warning.raise();
        }
      }
      // local function tasks have no region requirements so nothing below
      if (local_function)
        return;
      update_no_access_regions();
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
        log_requirement(unique_op_id, idx, logical_regions[idx]);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      analyze_region_requirements();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::perform_base_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // To be correct with the new scheduler we also have to
      // register mapping dependences on futures
      for (const Future& future : futures)
        if (future.impl != nullptr)
          future.impl->register_dependence(this);
      if (predicate_false_future.impl != nullptr)
        predicate_false_future.impl->register_dependence(this);
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(
            this, wait_barriers, arrive_barriers, must_epoch);
      version_infos.resize(logical_regions.size());
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Compute the parent region indexes since we might not have
      // computed them when we launched the task but we're going to
      // need them now
      if (trace != nullptr)
        compute_parent_indexes(true /*force*/);
      // Dumb case for must epoch operations, we need these to
      // be mapped immediately, mapper be damned
      if (must_epoch != nullptr)
      {
        TriggerTaskArgs trigger_args(this, parent_ctx->did);
        runtime->issue_runtime_meta_task(
            trigger_args, LG_THROUGHPUT_WORK_PRIORITY);
      }
      // If we're replaying this for for a trace then don't even
      // bother asking the mapper about when to map this
      else if (is_replaying() || local_function)
        enqueue_ready_operation();
      // Figure out whether this task is local or remote
      else if (!runtime->is_local(target_proc))
      {
        // We can only send it away if it is not origin mapped
        // otherwise it has to stay here until it is fully mapped
        if (!is_origin_mapped())
          runtime->send_task(this);
        else
          enqueue_ready_task(false /*use target*/);
      }
      else
        enqueue_ready_task(true /*use target*/);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::report_interfering_requirements(
        unsigned idx1, unsigned idx2)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx1 < idx2);
      // The logical dependence analysis can report this because there are
      // interfering fields and regions, check to make sure there are alos
      // interfering privileges and index spaces
      const RegionRequirement& req1 = get_requirement(idx1);
      const RegionRequirement& req2 = get_requirement(idx2);
      if (IS_READ_ONLY(req1) && IS_READ_ONLY(req2))
        return;
      if (IS_REDUCE(req1) && IS_REDUCE(req2) && (req1.redop == req2.redop))
        return;
      if (!runtime->are_disjoint(
              req1.region.get_index_space(), req2.region.get_index_space()))
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Found aliasing region requirements " << idx1 << " and "
              << idx2 << " of " << *this
              << ". Individual task launches are not permitted "
              << " to have interfering region requirements.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::record_trace_hash(
        TraceHashRecorder& recorder, uint64_t opidx)
    //--------------------------------------------------------------------------
    {
      if (output_regions.size() > 0)
        return recorder.record_operation_untraceable(this, opidx);
      Murmur3Hasher hasher;
      hasher.hash(get_operation_kind());
      hasher.hash(task_id);
      for (const RegionRequirement& req : regions)
        hash_requirement(hasher, req);
      hasher.hash<bool>(is_index_space);
      if (is_index_space)
      {
        hasher.hash<bool>(concurrent_task);
        hasher.hash<bool>(must_epoch_task);
        hasher.hash(index_domain);
      }
      if (future_return_size)
        hasher.hash(*future_return_size);
      return recorder.record_operation_hash(this, hasher, opidx);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      if (!elide_future_return)
      {
        // Set the future to the false result
        if (predicate_false_future.impl == nullptr)
        {
          if (predicate_false_result.get_size() > 0)
            result.impl->set_local(
                predicate_false_result.get_buffer(),
                predicate_false_result.get_size(), false /*own*/);
          else
            result.impl->set_result(ApEvent::NO_AP_EVENT, nullptr);
        }
        else
        {
          // Safe to block here indefinitely waiting for unbounded pools
          result.impl->set_result(
              this, predicate_false_future.impl,
              nullptr /*safe_for_unbounded_pools*/);
        }
      }
      complete_mapping();
      complete_execution();
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      if (is_origin_mapped())
      {
        if (!runtime->is_local(target_proc))
        {
          runtime->send_task(this);
          return false;
        }
        else
          return true;
      }
      else
      {
        if (target_proc.exists() && (target_proc != current_proc))
        {
          runtime->send_task(this);
          return false;
        }
        return true;
      }
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::perform_mapping(
        MustEpochOp* must_epoch_owner /*=nullptr*/,
        const DeferMappingArgs* args /* =nullptr*/)
    //--------------------------------------------------------------------------
    {
      if (!map_all_regions(must_epoch_owner, args))
        return false;
      // If we mapped, then we are no longer stealable
      stealable = false;
      // If we're remote, send back a message to the origin node instance
      // of the task to tell it that we are mapped
      if (is_remote())
      {
        IndividualRemoteMapped rez;
        {
          RezCheck z(rez);
          rez.serialize<SingleTask*>(orig_task);
          rez.serialize(get_mapped_event());
        }
        rez.dispatch(orig_proc.address_space());
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::finalize_map_task_output(
        Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
        MustEpochOp* must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      if (!SingleTask::finalize_map_task_output(
              input, output, must_epoch_owner))
        return false;
      if (concurrent_task)
      {
        legion_assert(must_epoch_task);
        legion_assert(is_origin_mapped());
        legion_assert(concurrent_postcondition.exists());
        concurrent_precondition = Runtime::create_rt_user_event();
        must_epoch->rendezvous_concurrent_mapped(concurrent_precondition);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::replicate_task(void)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        // Pull these onto the stack since it's unsafe to read them after
        // we call replicate task and it goes off and does stuff
        SingleTask* original = orig_task;
        const Processor orig = orig_proc;
        const RtEvent event = get_mapped_event();
        const bool result = SingleTask::replicate_task();
        if (result)
        {
          IndividualRemoteMapped rez;
          {
            RezCheck z(rez);
            rez.serialize<SingleTask*>(original);
            rez.serialize(event);
          }
          rez.dispatch(orig.address_space());
        }
        return result;
      }
      else
        return SingleTask::replicate_task();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_future_size(
        size_t return_type_size, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(!elide_future_return);
      if (is_remote())
      {
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        IndividualRemoteFutureSize rez;
        {
          RezCheck z(rez);
          rez.serialize(orig_task);
          rez.serialize(return_type_size);
          rez.serialize(done_event);
        }
        rez.dispatch(orig_proc.address_space());
        applied_events.insert(done_event);
      }
      else
        result.impl->set_future_result_size(
            return_type_size, runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::record_output_registered(
        RtEvent registered, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(registered.exists());
      if (is_remote())
      {
        // Send the message on to the origin node to tell it
        // to launch the meta task to perform the registration
        const RtUserEvent applied = Runtime::create_rt_user_event();
        IndividualRemoteOutputRegistration rez;
        {
          RezCheck z(rez);
          rez.serialize(orig_task);
          rez.serialize(registered);
          rez.serialize(applied);
        }
        rez.dispatch(orig_proc.address_space());
        applied_events.insert(applied);
      }
      else
      {
        // Launch the meta-task to perform the registration
        // Make sure we don't complete the task until this is done
        FinalizeOutputEqKDTreeArgs args(this);
        output_regions_registered = runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, registered);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualRemoteOutputRegistration::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndividualTask* task;
      derez.deserialize(task);
      RtEvent registered;
      derez.deserialize(registered);
      RtUserEvent applied;
      derez.deserialize(applied);
      std::set<RtEvent> applied_events;
      task->record_output_registered(registered, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(applied, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(applied);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTaskConcurrentRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndividualTask *local, *remote;
      derez.deserialize(local);
      derez.deserialize(remote);
      uint64_t lamport_clock;
      derez.deserialize(lamport_clock);
      bool poisoned;
      derez.deserialize(poisoned);
      local->get_must_epoch_op()->concurrent_allreduce(
          remote, source, lamport_clock, poisoned);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTaskConcurrentResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndividualTask* local;
      derez.deserialize(local);
      uint64_t lamport_clock;
      derez.deserialize(lamport_clock);
      bool poisoned;
      derez.deserialize(poisoned);
      local->finish_concurrent_allreduce(lamport_clock, poisoned);
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_origin) && stealable);
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_output_valid(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return output_region_options[idx].valid_requirement();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::is_output_grouped(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return output_region_options[idx].grouped_fields();
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind IndividualTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INDIVIDUAL_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // Invalidate the logical context so child operations that still have
      // mapping references can begin committing
      if (execution_context != nullptr)
        execution_context->invalidate_logical_context();
      if (is_remote())
      {
        IndividualRemoteComplete rez;
        pack_remote_complete(rez, effects);
        rez.dispatch(orig_proc.address_space());
        complete_operation(effects);
      }
      else if (must_epoch != nullptr)
      {
        must_epoch->notify_subop_complete(this, effects);
        complete_operation(effects);
      }
      else
        complete_operation(effects);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      // Invalidate any state that we had if we didn't already
      // Do this before sending the complete message to avoid the
      // race condition in the remote case where the top-level
      // context cleans on the owner node while we still need it
      std::set<RtEvent> commit_preconditions;
      if (execution_context != nullptr)
      {
        execution_context->invalidate_region_tree_contexts(
            is_top_level_task(), commit_preconditions);
        execution_context->log_created_requirements();
      }
      if (profiling_reported.exists())
      {
        finalize_single_task_profiling();
        commit_preconditions.insert(profiling_reported);
      }
      if (output_regions_registered.exists())
        commit_preconditions.insert(output_regions_registered);
      if (remote_commit_precondition.exists())
        commit_preconditions.insert(remote_commit_precondition);
      // For remote cases we have to keep track of the events for
      // returning any created logical state, we can't commit until
      // it is returned or we might prematurely release the references
      // that we hold on the version state objects
      // Pass back our created and deleted operations
      if ((execution_context != nullptr) && !is_remote())
      {
        if (top_level_task)
          execution_context->report_leaks_and_duplicates(commit_preconditions);
        else if (must_epoch != nullptr)
          execution_context->return_resources(
              must_epoch, context_index, commit_preconditions);
        else
          execution_context->return_resources(
              parent_ctx, context_index, commit_preconditions);
      }
      RtEvent commit_precondition;
      if (!commit_preconditions.empty())
        commit_precondition = Runtime::merge_events(commit_preconditions);
      if (is_remote())
      {
        IndividualRemoteCommit rez;
        pack_remote_commit(rez, commit_precondition);
        rez.dispatch(orig_proc.address_space());
      }
      if (must_epoch != nullptr)
      {
        must_epoch->notify_subop_commit(this, commit_precondition);
        commit_operation(true /*deactivate*/, commit_precondition);
      }
      else
        commit_operation(true /*deactivate*/, commit_precondition);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_future(
        ApEvent effects, FutureInstance* instance, const void* metadata,
        size_t metasize, FutureFunctor* functor, Processor future_proc,
        bool own_functor)
    //--------------------------------------------------------------------------
    {
      if (functor != nullptr)
      {
        legion_assert(instance == nullptr);
        legion_assert(metadata == nullptr);
        if (elide_future_return)
        {
          functor->callback_release_future();
          if (own_functor)
            delete functor;
        }
        else
          result.impl->set_result(effects, functor, own_functor, future_proc);
      }
      else
      {
        if ((instance != nullptr) && (instance->size > 0) &&
            (shard_manager == nullptr))
          check_future_return_bounds(instance);
        if (elide_future_return)
        {
          if ((instance != nullptr) && !instance->defer_deletion(effects))
            delete instance;
        }
        else
        {
          if ((instance != nullptr) && (instance->size > 0) &&
              (shard_manager == nullptr))
            check_future_return_bounds(instance);
          result.impl->set_result(effects, instance, metadata, metasize);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_mispredication(void)
    //--------------------------------------------------------------------------
    {
      // First thing: increment the meta-task counts since we decremented
      // them in case we didn't end up running
      runtime->increment_total_outstanding_tasks(
          MispredicationTaskArgs::TASK_ID, true /*meta*/);
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
      runtime->outstanding_counts[MispredicationTaskArgs::TASK_ID].fetch_add(1);
#endif
      if (!elide_future_return)
      {
        // Set the future to the false result
        if (predicate_false_future.impl == nullptr)
        {
          if (predicate_false_result.get_size() > 0)
            result.impl->set_local(
                predicate_false_result.get_buffer(),
                predicate_false_result.get_size(), false /*own*/);
          else
            result.impl->set_result(ApEvent::NO_AP_EVENT, nullptr);
        }
        else
          result.impl->set_result(
              execution_context, predicate_false_future.impl);
      }
      // Pretend like we executed the task
      execution_context->handle_mispredication();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::set_concurrent_postcondition(RtEvent postcondition)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch_task);
      legion_assert(!concurrent_postcondition.exists());
      concurrent_postcondition = postcondition;
    }

    //--------------------------------------------------------------------------
    uint64_t IndividualTask::order_collectively_mapped_unbounded_pools(
        uint64_t lamport_clock, bool need_result)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch_task);
      legion_assert(is_origin_mapped());
      return must_epoch->collective_lamport_allreduce(
          lamport_clock, need_result);
    }

    //--------------------------------------------------------------------------
    ApEvent IndividualTask::order_concurrent_launch(
        ApEvent start, VariantImpl* impl)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch_task);
      legion_assert(target_processors.size() == 1);
      legion_assert(concurrent_postcondition.exists());
      // See the comment in PointTask::order_concurrent_launch that
      // describes what we are doing here
      const OrderConcurrentLaunchArgs args(
          this, target_processors.front(), start, selected_variant);
      RtEvent precondition;
      if (start.exists())
        precondition = Runtime::protect_event(start);
      Runtime::trigger_event(concurrent_precondition, precondition);
      // Give this very high priority as it is likely on the critical path
      runtime->issue_runtime_meta_task(
          args, LG_RESOURCE_PRIORITY, concurrent_postcondition);
      return args.ready;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::concurrent_allreduce(
        ProcessorManager* manager, uint64_t lamport_clock, VariantID vid,
        bool poisoned)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch_task);
      legion_assert(manager->local_proc == target_proc);
      if (is_remote())
      {
        IndividualTaskConcurrentRequest rez;
        {
          RezCheck z(rez);
          rez.serialize(orig_task);
          rez.serialize(this);
          rez.serialize(lamport_clock);
          rez.serialize(poisoned);
        }
        rez.dispatch(orig_proc.address_space());
      }
      else
        must_epoch->concurrent_allreduce(
            this, runtime->address_space, lamport_clock, poisoned);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::finish_concurrent_allreduce(
        uint64_t lamport_clock, bool poisoned)
    //--------------------------------------------------------------------------
    {
      ProcessorManager* manager = runtime->find_processor_manager(target_proc);
      manager->finalize_concurrent_task_order(this, lamport_clock, poisoned);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::perform_concurrent_task_barrier(void)
    //--------------------------------------------------------------------------
    {
      // No-op
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::send_task(
        Processor target, std::vector<SingleTask*>& others)
    //--------------------------------------------------------------------------
    {
      TaskMessage rez;
      bool deactivate;
      {
        RezCheck z(rez);
        rez.serialize(target);
        rez.serialize(INDIVIDUAL_TASK_KIND);
        deactivate = pack_task(rez, target.address_space());
      }
      rez.dispatch(target.address_space());
      return deactivate;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::pack_task(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // Check to see if we are stealable, if not and we have not
      // yet been sent remotely, then send the state now
      RezCheck z(rez);
      parent_ctx->pack_inner_context(rez);
      pack_single_task(rez, target);
      rez.serialize<size_t>(output_region_options.size());
      for (const OutputOptions& option : output_region_options)
        rez.serialize(option);
      rez.serialize(orig_task);
      rez.serialize(remote_unique_id);
      rez.serialize(top_level_task);
      if (!elide_future_return)
      {
        result.impl->pack_future(rez, target);
        if (predicate_false_future.impl != nullptr)
          predicate_false_future.impl->pack_future(rez, target);
        else
          rez.serialize<DistributedID>(0);
        rez.serialize<size_t>(predicate_false_result.get_size());
        if (predicate_false_result.get_size() > 0)
          rez.serialize(
              predicate_false_result.get_buffer(),
              predicate_false_result.get_size());
      }
      if (must_epoch_task)
      {
        rez.serialize(concurrent_precondition);
        rez.serialize(concurrent_postcondition);
      }
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      // Mark that we sent this task remotely
      sent_remotely = true;
      // If this task is remote, then deactivate it, otherwise
      // we're local so we don't want to be deactivated for when
      // return messages get sent back.
      return is_remote();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::unpack_task(
        Deserializer& derez, Processor current, AddressSpaceID,
        std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      // Figure out what our parent context is
      parent_ctx = InnerContext::unpack_inner_context(derez);
      unpack_single_task(derez, ready_events);
      size_t output_regions_size = 0;
      derez.deserialize(output_regions_size);
      output_region_options.resize(output_regions_size);
      for (unsigned idx = 0; idx < output_regions_size; idx++)
        derez.deserialize(output_region_options[idx]);
      derez.deserialize(orig_task);
      derez.deserialize(remote_unique_id);
      set_current_proc(current);
      derez.deserialize(top_level_task);
      // Quick check to see if we've been sent back to our original node
      if (!is_remote())
      {
#ifdef LEGION_DEBUG
        // Need to make the deserializer happy in debug mode
        // 2 *sizeof(size_t) since we're two DerezChecks deep
        derez.advance_pointer(derez.get_remaining_bytes() - 2 * sizeof(size_t));
#endif
        // If we were sent back then mark that we are no longer remote
        orig_task->sent_remotely = false;
        // Put the original instance back on the mapping queue and
        // deactivate this version of the task
        orig_task->enqueue_ready_task(false /*target*/);
        deactivate();
        return false;
      }
      if (!elide_future_return)
      {
        result = FutureImpl::unpack_future(derez);
        // Unpack the predicate false infos
        predicate_false_future = FutureImpl::unpack_future(derez);
        size_t predicate_false_size;
        derez.deserialize(predicate_false_size);
        if (predicate_false_size > 0)
        {
          predicate_false_result.save_buffer(
              derez.get_current_pointer(), predicate_false_size);
          derez.advance_pointer(predicate_false_size);
        }
      }
      if (must_epoch_task)
      {
        derez.deserialize(concurrent_precondition);
        derez.deserialize(concurrent_postcondition);
      }
      if (is_origin_mapped())
      {
        if (!is_leaf())
        {
          // Send back the event that will be triggered when the task is mapped
          IndividualRemoteMapped rez;
          {
            RezCheck z2(rez);
            rez.serialize<SingleTask*>(orig_task);
            rez.serialize(get_mapped_event());
          }
          rez.dispatch(orig_proc.address_space());
        }
        else
          // We're not going to get a callback from the context if we're a leaf
          complete_mapping();
      }
      else
        version_infos.resize(logical_regions.size());
      set_provenance(Provenance::deserialize(derez), true /*has ref*/);
      // Set our parent task for the user
      parent_task = parent_ctx->get_task();
      // Have to do this before resolving speculation in case
      // we get cleaned up after the resolve speculation call
      LegionSpy::log_point_point(remote_unique_id, get_unique_id());
      if (implicit_profiler != nullptr)
        implicit_profiler->register_operation(this);
      // Return true to add ourselves to the ready queue
      return true;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_complete(Serializer& rez, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // Send back the pointer to the task instance, then serialize
      // everything else that needs to be sent back
      rez.serialize(orig_task);
      rez.serialize(effects);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_complete(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      ApEvent effect;
      derez.deserialize(effect);
      record_completion_effect(effect);
      // Mark that we have both finished executing
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_commit(Serializer& rez, RtEvent pre)
    //--------------------------------------------------------------------------
    {
      // Only need to send back the pointer to the task instance
      rez.serialize(orig_task);
      RezCheck z(rez);
      rez.serialize(pre);
      // Pack the privilege state
      if (execution_context != nullptr)
      {
        rez.serialize<bool>(true);
        execution_context->pack_return_resources(rez, context_index);
      }
      else
        rez.serialize<bool>(false);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_commit(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtEvent remote_commit_precondition;
      derez.deserialize(remote_commit_precondition);
      // First unpack the privilege state
      bool has_privilege_state;
      derez.deserialize(has_privilege_state);
      if (has_privilege_state)
      {
        const RtEvent resources_returned =
            (must_epoch == nullptr) ?
                ResourceTracker::unpack_resources_return(derez, parent_ctx) :
                ResourceTracker::unpack_resources_return(derez, must_epoch);
        if (resources_returned.exists())
        {
          if (remote_commit_precondition.exists())
            remote_commit_precondition = Runtime::merge_events(
                remote_commit_precondition, resources_returned);
          else
            remote_commit_precondition = resources_returned;
        }
      }
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::complete_replay(ApEvent instance_ready_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(!target_processors.empty());
      const AddressSpaceID target_space =
          target_processors.front().address_space();
      // Check to see if we're replaying this locally or remotely
      if (target_space != runtime->address_space)
      {
        // This is the remote case, pack it up and ship it over
        // Mark that we are effecitvely mapping this at the origin
        map_origin = true;
        // Pack this task up and send it to the remote node
        RemoteTaskReplay rez;
        {
          RezCheck z(rez);
          rez.serialize(instance_ready_event);
          rez.serialize(target_processors.front());
          rez.serialize(INDIVIDUAL_TASK_KIND);
          pack_task(rez, target_space);
        }
        rez.dispatch(target_space);
      }
      else
      {
        legion_assert(is_leaf());
        legion_assert(region_preconditions.empty());
        region_preconditions.resize(regions.size(), instance_ready_event);
        execution_fence_event = instance_ready_event;
        update_no_access_regions();
        launch_task();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualRemoteFutureSize::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndividualTask* task;
      derez.deserialize(task);
      size_t return_type_size;
      derez.deserialize(return_type_size);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      std::set<RtEvent> applied_events;
      task->handle_future_size(return_type_size, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(
            done_event, Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualRemoteMapped::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      // Single task because we share this with the point task for when
      // point tasks are origin-mapped non-leaf tasks
      SingleTask* task;
      derez.deserialize(task);
      RtEvent mapped_event;
      derez.deserialize(mapped_event);
      task->handle_post_mapped(mapped_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualRemoteComplete::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      IndividualTask* task;
      derez.deserialize(task);
      task->unpack_remote_complete(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualRemoteCommit::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      IndividualTask* task;
      derez.deserialize(task);
      task->unpack_remote_commit(derez);
    }

    /////////////////////////////////////////////////////////////
    // Repl Individual Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndividualTask::ReplIndividualTask(void) : IndividualTask()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplIndividualTask::~ReplIndividualTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::activate(void)
    //--------------------------------------------------------------------------
    {
      IndividualTask::activate();
      owner_shard = 0;
      launch_space = nullptr;
      sharding_functor = std::numeric_limits<ShardingID>::max();
      sharding_function = nullptr;
      output_bar = RtBarrier::NO_RT_BARRIER;
      sharding_collective = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (sharding_collective != nullptr)
        delete sharding_collective;
      IndividualTask::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // We might be able to skip this if the sharding function was already
      // picked for us which occurs when we're part of a must-epoch launch
      if (sharding_function == nullptr)
      {
        // Do the mapper call to get the sharding function to use
        if (mapper == nullptr)
          mapper = runtime->find_mapper(current_proc, map_id);
        Mapper::SelectShardingFunctorInput* input = repl_ctx->shard_manager;
        Mapper::SelectShardingFunctorOutput output;
        mapper->invoke_task_select_sharding_functor(this, *input, output);
        if (output.chosen_functor == std::numeric_limits<ShardingID>::max())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << mapper->get_mapper_name()
                << " failed to pick a valid sharding functor for "
                << "task " << get_task_name() << " (UID " << get_unique_id()
                << ").";
          error.raise();
        }
        this->sharding_functor = output.chosen_functor;
        sharding_function =
            repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      }
      legion_assert(sharding_function != nullptr);
      // In debug mode we check to make sure that all the mappers
      // picked the same sharding function
      // Contribute the result
      if (runtime->safe_mapper)
      {
        legion_assert(sharding_collective != nullptr);
        sharding_collective->contribute(this->sharding_functor);
        if (sharding_collective->is_target() &&
            !sharding_collective->validate(this->sharding_functor))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << mapper->get_mapper_name()
                << " chose different sharding functions "
                << "for individual task " << get_task_name() << " (UID "
                << get_unique_id() << ") in " << parent_ctx->get_task_name()
                << " (UID " << parent_ctx->get_unique_id() << ").";
          error.raise();
        }
      }
      // Now we can do the normal prepipeline stage
      IndividualTask::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      LogicalAnalysis logical_analysis(this, regions.size());
      ShardingFunction* analysis_sharding_function = sharding_function;
      if (must_epoch_task)
      {
        // Note we use a special
        // projection function for must epoch launches that maps all the
        // tasks to the special shard UINT_MAX so that they appear to be
        // on a different shard than any other tasks, but on the same shard
        // for all the tasks in the must epoch launch.
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        analysis_sharding_function =
            repl_ctx->get_universal_sharding_function();
      }

      analyze_region_requirements(
          launch_space, analysis_sharding_function, sharding_space);
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(sharding_function != nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Figure out whether this shard owns this point
      if (sharding_space.exists())
      {
        Domain shard_domain;
        runtime->find_domain(sharding_space, shard_domain);
        owner_shard = sharding_function->find_owner(index_point, shard_domain);
      }
      else
        owner_shard = sharding_function->find_owner(index_point, index_domain);
      // If we're recording then record the owner shard
      if (is_recording())
      {
        legion_assert(!is_remote());
        legion_assert((tpl != nullptr) && tpl->is_recording());
        tpl->record_owner_shard(trace_local_id, owner_shard);
      }
      LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
      // If we own it we go on the queue, otherwise we complete early
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        if (output_bar.exists())
          record_output_registered(
              RtEvent::NO_RT_EVENT, map_applied_conditions);
        shard_off(RtEvent::NO_RT_EVENT);
      }
      else  // We own it, so it goes on the ready queue
      {
        // Don't signal the tree yet, we need to wait to see how big
        // the result future size is first
        // Then we can do the normal analysis
        IndividualTask::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      // Figure out if we're the one to do the replay
      legion_assert(!is_remote());
      legion_assert(tpl != nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      owner_shard = tpl->find_owner_shard(trace_local_id);
      LegionSpy::log_owner_shard(get_unique_id(), owner_shard);
      if (owner_shard != repl_ctx->owner_shard->shard_id)
      {
        LegionSpy::log_replay_operation(unique_op_id);
        shard_off(RtEvent::NO_RT_EVENT);
      }
      else
        IndividualTask::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Only set the future on shard 0 (note we know that all the shards
      // have resolved false so we don't need to ask the sharding functor
      // which one we want to do the work)
      // Trigger the output barrier if we have one
      if (output_bar.exists())
        runtime->phase_barrier_arrive(output_bar, 1 /*count*/);
      if (repl_ctx->owner_shard->shard_id > 0)
        shard_off(RtEvent::NO_RT_EVENT);
      else
        IndividualTask::predicate_false();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::shard_off(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
      // Still need this to record that this operation is done for LegionSpy
      LegionSpy::log_operation_events(
          unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
      complete_mapping(mapped_precondition);
      complete_execution();
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::prepare_map_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch != nullptr);
      legion_assert(sharding_function != nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      set_origin_mapped(true);
      // See if we're going to be a local point or not
      Domain shard_domain = index_domain;
      if (sharding_space.exists())
        runtime->find_domain(sharding_space, shard_domain);
      if (!elide_future_return)
      {
        ShardID owner =
            sharding_function->find_owner(index_point, shard_domain);
        if (owner == repl_ctx->owner_shard->shard_id)
        {
          FutureMap map = must_epoch->get_future_map();
          result = map.impl->get_future(index_point, true /*internal only*/);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      IndexSpace handle;
      if (index_domain.get_dim() == 0)
      {
        DomainPoint point(0);
        Domain launch_domain(point, point);
        handle = ctx->find_index_launch_space(launch_domain, get_provenance());
      }
      else
        handle = ctx->find_index_launch_space(index_domain, get_provenance());
      launch_space = runtime->get_node(handle);
      if (!output_regions.empty())
        output_bar = ctx->get_next_output_regions_barrier();
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::set_sharding_function(
        ShardingID functor, ShardingFunction* function)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch != nullptr);
      legion_assert(sharding_function == nullptr);
      sharding_functor = functor;
      sharding_function = function;
    }

    //--------------------------------------------------------------------------
    Future ReplIndividualTask::create_future(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      DistributedID future_did = repl_ctx->get_next_distributed_id();
      return repl_ctx->shard_manager->deduplicate_future_creation(
          repl_ctx, future_did, this, index_point);
    }

    //--------------------------------------------------------------------------
    void ReplIndividualTask::record_output_registered(
        RtEvent registered, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_remote());
      legion_assert(output_bar.exists());
      legion_assert(!output_regions.empty());
      // Launch the meta-task to perform the registration
      // Make sure we don't complete the task until the barrier is done
      // on the shard that actually owns the task
      runtime->phase_barrier_arrive(output_bar, 1 /*count*/, registered);
      FinalizeOutputEqKDTreeArgs args(this);
      output_regions_registered = runtime->issue_runtime_meta_task(
          args, LG_LATENCY_DEFERRED_PRIORITY, output_bar);
    }

  }  // namespace Internal
}  // namespace Legion
