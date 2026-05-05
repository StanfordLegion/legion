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

#include "legion/tasks/task.h"
#include "legion/contexts/inner.h"
#include "legion/api/functors_impl.h"
#include "legion/api/future_impl.h"
#include "legion/api/physical_region_impl.h"
#include "legion/managers/mapper.h"
#include "legion/nodes/field.h"
#include "legion/tasks/individual.h"
#include "legion/tasks/slice.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // External Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalTask::ExternalTask(void) : Task()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ExternalTask::pack_external_task(
        Serializer& rez, AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(task_id);
      rez.serialize(indexes.size());
      for (const IndexSpaceRequirement& index : indexes)
        pack_index_space_requirement(index, rez);
      rez.serialize(regions.size());
      for (const RegionRequirement& region : regions)
        pack_region_requirement(region, rez);
      rez.serialize(output_regions.size());
      for (const OutputRequirement& output : output_regions)
        pack_output_requirement(output, rez);
      rez.serialize(futures.size());
      // If we are remote we can just do the normal pack
      for (const Future& future : futures)
        if (future.impl != nullptr)
          future.impl->pack_future(rez, target);
        else
          rez.serialize<DistributedID>(0);
      rez.serialize(grants.size());
      for (const Grant& grant : grants) pack_grant(grant, rez);
      rez.serialize(wait_barriers.size());
      for (const PhaseBarrier& barrier : wait_barriers)
        pack_phase_barrier(barrier, rez);
      rez.serialize(arrive_barriers.size());
      for (const PhaseBarrier& barrier : arrive_barriers)
        pack_phase_barrier(barrier, rez);
      arg_manager.serialize(rez);
      // Not the argmanager here in case the buffer is from a future
      rez.serialize(local_arglen);
      if (local_arglen)
        rez.serialize(local_args, local_arglen);
      pack_mappable(*this, rez);
      rez.serialize(is_index_space);
      rez.serialize(concurrent_task);
      rez.serialize(must_epoch_task);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      rez.serialize(sharding_space);
      rez.serialize(orig_proc);
      // No need to pack current proc, it will get set when we unpack
      rez.serialize(steal_count);
      // No need to pack remote, it will get set
      rez.serialize(speculated);
      // No need to pack local function, it's not if we're sending this remote
      rez.serialize<uint64_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalTask::unpack_external_task(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(task_id);
      size_t num_indexes;
      derez.deserialize(num_indexes);
      indexes.resize(num_indexes);
      for (IndexSpaceRequirement& index : indexes)
        unpack_index_space_requirement(index, derez);
      size_t num_regions;
      derez.deserialize(num_regions);
      regions.resize(num_regions);
      for (RegionRequirement& region : regions)
        unpack_region_requirement(region, derez);
      size_t num_output_regions;
      derez.deserialize(num_output_regions);
      output_regions.resize(num_output_regions);
      for (OutputRequirement& output : output_regions)
        unpack_output_requirement(output, derez);
      size_t num_futures;
      derez.deserialize(num_futures);
      futures.resize(num_futures);
      for (unsigned idx = 0; idx < futures.size(); idx++)
        futures[idx] = FutureImpl::unpack_future(derez);
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (Grant& grant : grants) unpack_grant(grant, derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (PhaseBarrier& barrier : wait_barriers)
        unpack_phase_barrier(barrier, derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (PhaseBarrier& barrier : arrive_barriers)
        unpack_phase_barrier(barrier, derez);
      arg_manager.deserialize(derez);
      arglen = arg_manager.get_size();
      args = arg_manager.get_buffer();
      local_arg_manager.deserialize(derez);
      local_arglen = local_arg_manager.get_size();
      local_args = local_arg_manager.get_buffer();
      unpack_mappable(*this, derez);
      derez.deserialize(is_index_space);
      derez.deserialize(concurrent_task);
      derez.deserialize(must_epoch_task);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      derez.deserialize(sharding_space);
      derez.deserialize(orig_proc);
      derez.deserialize(steal_count);
      derez.deserialize(speculated);
      uint64_t index;
      derez.deserialize(index);
      set_context_index(index);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalTask::pack_output_requirement(
        const OutputRequirement& req, Serializer& rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_region_requirement(req, rez);
      rez.serialize(req.type_tag);
      rez.serialize(req.field_space);
      rez.serialize(req.global_indexing);
      rez.serialize(req.bounded_requirement);
      rez.serialize(req.color_space);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalTask::unpack_output_requirement(
        OutputRequirement& req, Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_region_requirement(req, derez);
      derez.deserialize(req.type_tag);
      derez.deserialize(req.field_space);
      derez.deserialize(req.global_indexing);
      derez.deserialize(req.bounded_requirement);
      derez.deserialize(req.color_space);
    }

    /////////////////////////////////////////////////////////////
    // Task Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskOp::TaskOp(void)
      : ExternalTask(), PredicatedOp(), logical_regions(TaskRequirements(*this))
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    TaskOp::~TaskOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID TaskOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t TaskOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void TaskOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int TaskOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(parent_ctx != nullptr);
      return parent_ctx->get_depth() + 1;
    }

    //--------------------------------------------------------------------------
    bool TaskOp::has_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return (get_depth() > 0);
    }

    //--------------------------------------------------------------------------
    const Task* TaskOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const char* TaskOp::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      TaskImpl* impl = runtime->find_or_create_task_impl(task_id);
      return impl->get_name();
    }

    //--------------------------------------------------------------------------
    bool TaskOp::is_reducing_future(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_remote_operation(
        Serializer& rez, AddressSpaceID target,
        std::set<RtEvent>& applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_task(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_profiling_requests(
        Serializer& rez, std::set<RtEvent>& applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(0);
      rez.serialize<size_t>(0);
    }

    //--------------------------------------------------------------------------
    bool TaskOp::is_remote(void) const
    //--------------------------------------------------------------------------
    {
      if (local_cached)
        return !is_local;
      if (!orig_proc.exists())
        is_local = runtime->is_local(parent_ctx->get_executing_processor());
      else
        is_local = runtime->is_local(orig_proc);
      local_cached = true;
      return !is_local;
    }

    //--------------------------------------------------------------------------
    bool TaskOp::is_forward_progress_task(void)
    //--------------------------------------------------------------------------
    {
      // A forward progress task is any task that needs to have some or all
      // of its point tasks mapped in order to avoid blocking the mapping
      // of other point tasks. This includes index space task launches with
      // collective mapping region requirements, or concurrent index space
      // task launches. Dependent index space task launches used to have
      // this property but no longer now that we fixed them so that we
      // enqueue the points in the ready queue in dependence order
      return (concurrent_task || !check_collective_regions.empty());
    }

    //--------------------------------------------------------------------------
    void TaskOp::set_current_proc(Processor current)
    //--------------------------------------------------------------------------
    {
      legion_assert(current.exists());
      legion_assert(runtime->is_local(current));
      // Always clear target_proc and the mapper when setting a new current proc
      mapper = nullptr;
      current_proc = current;
      target_proc = current;
    }

    //--------------------------------------------------------------------------
    void TaskOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PredicatedOp::activate();
      commit_received = false;
      children_commit = false;
      stealable = false;
      options_selected = false;
      map_origin = false;
      request_valid_instances = false;
      elide_future_return = false;
      replicate = false;
      local_cached = false;
      target_proc = Processor::NO_PROC;
      mapper = nullptr;
      must_epoch = nullptr;
      must_epoch_task = false;
      concurrent_task = false;
      local_function = false;
      orig_proc = Processor::NO_PROC;  // for is_remote
    }

    //--------------------------------------------------------------------------
    void TaskOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      indexes.clear();
      regions.clear();
      output_regions.clear();
      futures.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      arg_manager.clear();
      if (args != nullptr)
      {
        args = nullptr;
        arglen = 0;
      }
      // We don't own these so just reset them
      local_args = nullptr;
      local_arglen = 0;
      if (mapper_data != nullptr)
      {
        free(mapper_data);
        mapper_data = nullptr;
        mapper_data_size = 0;
      }
      check_collective_regions.clear();
      atomic_locks.clear();
      parent_req_indexes.clear();
      version_infos.clear();
      future_return_size.reset();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      PredicatedOp::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    void TaskOp::set_must_epoch(
        MustEpochOp* epoch, unsigned index, bool do_registration)
    //--------------------------------------------------------------------------
    {
      Operation::set_must_epoch(epoch, do_registration);
      must_epoch_index = index;
      must_epoch_task = true;
      concurrent_task = true;
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        const TaskKind kind = get_task_kind();
        if (kind == INDEX_TASK_KIND)
          LegionSpy::log_index_task(
              parent_ctx->get_unique_id(), unique_op_id, task_id,
              get_task_name());
        else if (kind == INDIVIDUAL_TASK_KIND)
          LegionSpy::log_individual_task(
              parent_ctx->get_unique_id(), unique_op_id, task_id,
              get_task_name());
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_base_task(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // pack all the user facing data first
      pack_external_task(rez, target);
      RezCheck z(rez);
      rez.serialize(parent_req_indexes.size());
      for (unsigned idx = 0; idx < parent_req_indexes.size(); idx++)
      {
        if (parent_req_indexes[idx] == TRACED_PARENT_INDEX)
          parent_req_indexes[idx] = parent_ctx->find_parent_region_index(
              this, logical_regions[idx], idx, false /*skip privilege*/,
              true /*force*/);
        rez.serialize(parent_req_indexes[idx]);
      }
      rez.serialize(memo_state);
      rez.serialize(map_origin);
      if (map_origin)
      {
        rez.serialize<size_t>(atomic_locks.size());
        for (const std::pair<const Reservation, bool>& lock_pair : atomic_locks)
        {
          rez.serialize(lock_pair.first);
          rez.serialize(lock_pair.second);
        }
      }
      else
      {
        if (memo_state == MEMO_RECORD)
        {
          rez.serialize(tpl);
          rez.serialize(trace_local_id);
        }
        rez.serialize<size_t>(check_collective_regions.size());
        for (unsigned region_idx : check_collective_regions)
          rez.serialize(region_idx);
      }
      rez.serialize(future_return_size);
      rez.serialize(request_valid_instances);
      rez.serialize(execution_fence_event);
      rez.serialize(elide_future_return);
      rez.serialize(replicate);
      rez.serialize(true_guard);
      rez.serialize(false_guard);
      rez.serialize(exception_handler);
    }

    //--------------------------------------------------------------------------
    void TaskOp::unpack_base_task(
        Deserializer& derez, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      // unpack all the user facing data
      unpack_external_task(derez);
      DerezCheck z(derez);
      size_t num_indexes;
      derez.deserialize(num_indexes);
      if (num_indexes > 0)
      {
        parent_req_indexes.resize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
          derez.deserialize(parent_req_indexes[idx]);
      }
      derez.deserialize(memo_state);
      derez.deserialize(map_origin);
      if (map_origin)
      {
        size_t num_atomic;
        derez.deserialize(num_atomic);
        for (unsigned idx = 0; idx < num_atomic; idx++)
        {
          Reservation lock;
          derez.deserialize(lock);
          derez.deserialize(atomic_locks[lock]);
        }
      }
      else
      {
        if (memo_state == MEMO_RECORD)
        {
          derez.deserialize(tpl);
          derez.deserialize(trace_local_id);
        }
        size_t num_check_collective_regions;
        derez.deserialize(num_check_collective_regions);
        check_collective_regions.resize(num_check_collective_regions);
        for (unsigned idx = 0; idx < num_check_collective_regions; idx++)
          derez.deserialize(check_collective_regions[idx]);
      }
      derez.deserialize(future_return_size);
      derez.deserialize(request_valid_instances);
      derez.deserialize(execution_fence_event);
      derez.deserialize(elide_future_return);
      derez.deserialize(replicate);
      derez.deserialize(true_guard);
      derez.deserialize(false_guard);
      derez.deserialize(exception_handler);
      // Already had our options selected
      options_selected = true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskMessage::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // Figure out what kind of task this is and where it came from
      DerezCheck z(derez);
      Processor current;
      derez.deserialize(current);
      TaskOp::TaskKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case TaskOp::INDIVIDUAL_TASK_KIND:
          {
            IndividualTask* task = runtime->get_operation<IndividualTask>();
            std::set<RtEvent> ready_events;
            if (task->unpack_task(derez, current, source, ready_events))
            {
              RtEvent ready;
              if (!ready_events.empty())
                ready = Runtime::merge_events(ready_events);
              // Origin mapped tasks can go straight to launching
              // themselves since they are already mapped
              if (task->is_origin_mapped())
              {
                if (ready.exists() && !ready.has_triggered())
                {
                  TaskOp::TriggerTaskArgs trigger_args(
                      task, task->get_context()->did);
                  runtime->issue_runtime_meta_task(
                      trigger_args, LG_THROUGHPUT_WORK_PRIORITY, ready);
                }
                else
                  task->trigger_mapping();
              }
              else
                task->enqueue_ready_task(false /*target*/, ready);
            }
            break;
          }
        case TaskOp::SLICE_TASK_KIND:
          {
            SliceTask* task = runtime->get_operation<SliceTask>();
            std::set<RtEvent> ready_events;
            if (task->unpack_task(derez, current, source, ready_events))
            {
              RtEvent ready;
              if (!ready_events.empty())
                ready = Runtime::merge_events(ready_events);
              // Invoke trigger mapping on this slice
              if (ready.exists() && !ready.has_triggered())
              {
                TaskOp::TriggerTaskArgs trigger_args(
                    task, task->get_context()->did);
                runtime->issue_runtime_meta_task(
                    trigger_args, LG_THROUGHPUT_WORK_PRIORITY, ready);
              }
              else
                task->trigger_mapping();
            }
            break;
          }
        default:
          std::abort();  // no other tasks should be sent anywhere
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTaskReplay::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      // Figure out what kind of task this is and where it came from
      DerezCheck z(derez);
      ApEvent instance_ready;
      derez.deserialize(instance_ready);
      Processor target_proc;
      derez.deserialize(target_proc);
      TaskOp::TaskKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case TaskOp::INDIVIDUAL_TASK_KIND:
          {
            IndividualTask* task = runtime->get_operation<IndividualTask>();
            std::set<RtEvent> ready_events;
            task->unpack_task(derez, target_proc, source, ready_events);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            task->complete_replay(instance_ready);
            break;
          }
        case TaskOp::SLICE_TASK_KIND:
          {
            SliceTask* task = runtime->get_operation<SliceTask>();
            std::set<RtEvent> ready_events;
            task->unpack_task(derez, target_proc, source, ready_events);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            task->complete_replay(instance_ready);
            break;
          }
        default:
          std::abort();  // no other tasks should be sent anywhere
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::mark_stolen(void)
    //--------------------------------------------------------------------------
    {
      steal_count++;
    }

    //--------------------------------------------------------------------------
    void TaskOp::initialize_base_task(
        InnerContext* ctx, const Predicate& p, Processor::TaskFuncID tid,
        Provenance* prov)
    //--------------------------------------------------------------------------
    {
      initialize_predication(ctx, p, prov);
      parent_task = ctx->get_task();  // initialize the parent task
      // Fill in default values for all of the Task fields
      orig_proc = ctx->get_executing_processor();
      current_proc = orig_proc;
      steal_count = 0;
      speculated = false;
      local_function = false;
    }

    //--------------------------------------------------------------------------
    bool TaskOp::select_task_options(bool prioritize)
    //--------------------------------------------------------------------------
    {
      legion_assert(!options_selected);
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      Mapper::TaskOptions options;
      options.initial_proc = current_proc;
      options.valid_instances = mapper->request_valid_instances;
      const TaskPriority parent_priority =
          parent_ctx->is_priority_mutable() ?
              parent_ctx->get_current_priority() :
              0;
      options.parent_priority = parent_priority;
      mapper->invoke_select_task_options(this, options, prioritize);
      options_selected = true;
      if (options.initial_proc.kind() == Processor::UTIL_PROC)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output. Mapper " << *mapper
              << " requested that " << *this << " initially be assigned to a "
              << "utility processor in 'select_task_options.' Only application "
              << "processor kinds are permitted to be the target processor for "
                 "tasks.";
        error.raise();
      }
      target_proc = options.initial_proc;
      if (local_function && !runtime->is_local(target_proc))
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output. Mapper " << *mapper
              << "requested that local function " << *this
              << " be assigned to processor " << target_proc << " which is "
              << "not local to address space " << runtime->address_space
              << ". Local function tasks must be assigned to local processors.";
        error.raise();
      }
      stealable = options.stealable;
      map_origin = options.map_locally;
      request_valid_instances = options.valid_instances;
      if (parent_priority != options.parent_priority)
      {
        // Request for priority change see if it is legal or not
        if (parent_ctx->is_priority_mutable())
          parent_ctx->set_current_priority(options.parent_priority);
        else
        {
          Warning warning;
          warning << "Mapper " << *mapper
                  << " requested change of priority for parent "
                  << *(parent_ctx->owner_task) << " when launching " << *this
                  << " but the parent "
                  << "context does not support parent task priority mutation";
          warning.raise();
        }
      }
      if (is_index_space)
      {
        if (!options.check_collective_regions.empty())
        {
          for (const unsigned& region_idx : options.check_collective_regions)
          {
            if (region_idx >= regions.size())
              continue;
            const RegionRequirement& req = regions[region_idx];
            if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
              continue;
            if (!IS_WRITE(req))
              check_collective_regions.emplace_back(region_idx);
            else if (!IS_COLLECTIVE(req))
            {
              Warning warning;
              warning
                  << "Ignoring request by mapper " << *mapper
                  << " to check for collective usage for region requirement "
                  << region_idx << " of " << *this
                  << " because region requirement "
                  << "has writing privileges.";
              warning.raise();
            }
          }
        }
        for (unsigned idx = 0; idx < regions.size(); idx++)
          if (IS_COLLECTIVE(regions[idx]))
            check_collective_regions.emplace_back(idx);
        if (!check_collective_regions.empty())
        {
          std::sort(
              check_collective_regions.begin(), check_collective_regions.end());
          // Check to make sure that there are no invertible projection functors
          // in this index space launch on writing requirements which might
          // cause point tasks to be interfering. If there are then we can't
          // perform any collective rendezvous here so the tasks map together
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            const RegionRequirement& req = regions[idx];
            if (!IS_WRITE(req) || IS_COLLECTIVE(req))
              continue;
            if (((req.projection == 0) &&
                 (req.handle_type == LEGION_REGION_PROJECTION)) ||
                runtime->find_projection_function(req.projection)
                    ->is_invertible)
            {
              // Has potential dependences between the points so we can't
              // assume that this is safe
              check_collective_regions.clear();
              break;
            }
          }
        }
      }
      if (options.replicate)
      {
        // Replication of concurrent index space task launches are illegal
        if (concurrent_task)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << *mapper << " request to replicate " << *this
                << " that is a concurrent index space task launch in "
                << "'select_task_options'. It is illegal to replicate the "
                << "point tasks of a concurrent index space task launch.";
          error.raise();
        }
        // Replication of must epoch tasks are not allowed
        if (must_epoch_task)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << *mapper << " requested to replicate must epoch "
                << *this
                << ". Replication of must epoch tasks are not supported.";
          error.raise();
        }
        // Replication of origin-mapped tasks is not supported
        if (map_origin)
        {
          Warning warning;
          warning
              << "Mapper " << *mapper
              << " requested to both replicate and origin map " << *this
              << " in 'select_task_options'. Replication of origin-"
              << "mapped tasks is not currently supported and the request to "
              << "replicate the task will be ignored.";
          warning.raise();
          options.replicate = false;
        }
        // Output regions are not currently supported
        if (!output_regions.empty())
        {
          Warning warning;
          warning
              << "Mapper " << *mapper << " requested to replicate " << *this
              << " with output regions in 'select_task_options'. Legion does "
              << "not currently support replication of tasks with output "
              << "regions at the moment. You can request support for this "
              << "feature by emailing the the Legion developers list or "
              << "opening a github issue. The mapper call to replicate_task "
              << "is being elided.";
          warning.raise();
          options.replicate = false;
        }
        // We allow replication of tasks with reduction privileges, but
        // not if they are also part of a collective region requirement
        // because we don't know how to make a collective view that is
        // replicated for all the shards of the repdlicated task, but then
        // an all-reduce view across the points of the index space task
        // launch that are operating collectively
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
        {
          if (!IS_REDUCE(logical_regions[idx]))
            continue;
          Warning warning;
          warning
              << "Mapper " << *mapper << " requested to replicate " << *this
              << " with reduction privilege on region requirement " << idx
              << " in 'select_task_options'. Legion does not currently support "
              << "replication of tasks with reduction privileges. "
              << "You can request support for this feature by emailing the "
              << "Legion developers list or opening a github issue. The mapper "
              << "call to replicate_task is being elided.";
          warning.raise();
          options.replicate = false;
          break;
        }
        replicate = options.replicate;
      }
      if (options.inline_task)
      {
        if (concurrent_task)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << *mapper << " requested to inline concurrent "
                << *this << ". Inlining of concurrent tasks are not supported.";
          error.raise();
        }
        if (must_epoch_task)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << *mapper << " requested to inline must epoch "
                << *this << ". Inlining of must epoch tasks are not supported.";
          error.raise();
        }
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    const char* TaskOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return get_task_name();
    }

    //--------------------------------------------------------------------------
    OpKind TaskOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TASK_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t TaskOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      return logical_regions.size();
    }

    //--------------------------------------------------------------------------
    Mappable* TaskOp::get_mappable(void)
    //--------------------------------------------------------------------------
    {
      return this;
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool task_commit = false;
      {
        AutoLock o_lock(op_lock);
        legion_assert(!commit_received);
        commit_received = true;
        // If we already received the child commit then we
        // are ready to commit this task
        task_commit = children_commit;
      }
      if (task_commit)
        trigger_task_commit();
    }

    //--------------------------------------------------------------------------
    void TaskOp::select_sources(
        const unsigned index, PhysicalManager* target,
        const std::vector<InstanceView*>& sources,
        std::vector<unsigned>& ranking,
        std::map<unsigned, PhysicalManager*>& points)
    //--------------------------------------------------------------------------
    {
      legion_assert(index < regions.size());
      Mapper::SelectTaskSrcInput input;
      Mapper::SelectTaskSrcOutput output;
      prepare_for_mapping(target, input.target);
      prepare_for_mapping(
          sources, input.source_instances, input.collective_views);
      input.region_req_index = index;
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_select_task_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_atomic_locks(
        const unsigned index, Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::map<Reservation, bool>::iterator finder = atomic_locks.find(lock);
      if (finder != atomic_locks.end())
      {
        if (!finder->second && exclusive)
          finder->second = true;
      }
      else
        atomic_locks[lock] = exclusive;
    }

    //--------------------------------------------------------------------------
    unsigned TaskOp::find_parent_index(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < parent_req_indexes.size());
      legion_assert(parent_req_indexes[idx] != TRACED_PARENT_INDEX);
      return parent_req_indexes[idx];
    }

    //--------------------------------------------------------------------------
    VersionInfo& TaskOp::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < version_infos.size());
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    const VersionInfo& TaskOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < version_infos.size());
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>* TaskOp::get_acquired_instances_ref(
        void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    bool TaskOp::defer_perform_mapping(
        RtEvent precondition, MustEpochOp* op, unsigned invocation_count,
        std::vector<unsigned>* performed, std::vector<ApEvent>* effects)
    //--------------------------------------------------------------------------
    {
      DeferMappingArgs args(this, op, invocation_count, performed, effects);
      runtime->issue_runtime_meta_task(
          args, LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      return false;
    }

    //--------------------------------------------------------------------------
    void TaskOp::DeferMappingArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      implicit_operation = proxy_this;
      if (proxy_this->is_origin_mapped())
      {
        if (proxy_this->perform_mapping(must_op, this) &&
            proxy_this->distribute_task())
          proxy_this->launch_task();
      }
      else
      {
        if (proxy_this->perform_mapping(must_op, this))
          proxy_this->launch_task();
      }
    }

    //--------------------------------------------------------------------------
    const std::string_view& TaskOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    void TaskOp::activate_outstanding_task(void)
    //--------------------------------------------------------------------------
    {
      parent_ctx->increment_outstanding();
    }

    //--------------------------------------------------------------------------
    void TaskOp::deactivate_outstanding_task(void)
    //--------------------------------------------------------------------------
    {
      parent_ctx->decrement_outstanding();
    }

    //--------------------------------------------------------------------------
    void TaskOp::clone_task_op_from(TaskOp* rhs, Processor p, bool can_steal)
    //--------------------------------------------------------------------------
    {
      legion_assert(p.exists());
      // From Operation
      this->parent_ctx = rhs->parent_ctx;
      this->context_index = rhs->get_context_index();
      this->exception_handler = rhs->get_exception_handler();
      this->execution_fence_event = rhs->get_execution_fence_event();
      // Don't register this an operation when setting the must epoch info
      if (rhs->must_epoch != nullptr)
        this->set_must_epoch(
            rhs->must_epoch, rhs->must_epoch_index, false /*do registration*/);
      // From Memoizable
      this->trace_local_id = rhs->trace_local_id;
      // From Task
      this->task_id = rhs->task_id;
      this->indexes = rhs->indexes;
      this->regions = rhs->regions;
      this->output_regions = rhs->output_regions;
      this->futures = rhs->futures;
      this->grants = rhs->grants;
      this->wait_barriers = rhs->wait_barriers;
      this->arrive_barriers = rhs->arrive_barriers;
      this->arg_manager = rhs->arg_manager;
      this->arglen = this->arg_manager.get_size();
      this->args = this->arg_manager.get_buffer();
      this->map_id = rhs->map_id;
      this->tag = rhs->tag;
      if (rhs->mapper_data_size > 0)
      {
        legion_assert(rhs->mapper_data != nullptr);
        this->mapper_data_size = rhs->mapper_data_size;
        this->mapper_data = malloc(this->mapper_data_size);
        memcpy(this->mapper_data, rhs->mapper_data, this->mapper_data_size);
      }
      this->is_index_space = rhs->is_index_space;
      this->concurrent_task = rhs->concurrent_task;
      this->must_epoch_task = rhs->must_epoch_task;
      this->orig_proc = rhs->orig_proc;
      this->current_proc = rhs->current_proc;
      this->steal_count = rhs->steal_count;
      this->stealable = can_steal;
      this->speculated = rhs->speculated;
      this->parent_task = rhs->parent_task;
      this->map_origin = rhs->map_origin;
      this->elide_future_return = rhs->elide_future_return;
      this->replicate = rhs->replicate;
      this->sharding_space = rhs->sharding_space;
      this->request_valid_instances = rhs->request_valid_instances;
      this->future_return_size = rhs->future_return_size;
      // From TaskOp
      this->check_collective_regions = rhs->check_collective_regions;
      this->atomic_locks = rhs->atomic_locks;
      this->parent_req_indexes = rhs->parent_req_indexes;
      this->current_proc = rhs->current_proc;
      this->target_proc = p;
      this->true_guard = rhs->true_guard;
      this->false_guard = rhs->false_guard;
      // Memoizable stuff
      this->tpl = rhs->tpl;
      this->memo_state = rhs->memo_state;
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_grants(const std::vector<Grant>& requested_grants)
    //--------------------------------------------------------------------------
    {
      if (requested_grants.empty())
        return;
      grants = requested_grants;
      const ApEvent grant_pre = get_completion_event();
      for (const Grant& grant : grants)
        grant.impl->register_operation(grant_pre);
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_arrival_barriers(
        const std::vector<PhaseBarrier>& phase_barriers)
    //--------------------------------------------------------------------------
    {
      if (phase_barriers.empty())
        return;
      const ApEvent arrive_pre = get_completion_event();
      for (const PhaseBarrier& barrier : phase_barriers)
      {
        arrive_barriers.emplace_back(barrier);
        runtime->phase_barrier_arrive(barrier, 1 /*count*/, arrive_pre);
        LegionSpy::log_phase_barrier_arrival(
            unique_op_id, barrier.phase_barrier);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::finalize_output_region_trees(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!output_regions.empty());
      const size_t offset = regions.size();
      for (unsigned idx = 0; idx < output_regions.size(); idx++)
        if (!is_output_bounded(idx))
          parent_ctx->finalize_output_eqkd_tree(
              find_parent_index(offset + idx));
    }

    //--------------------------------------------------------------------------
    void TaskOp::validate_variant_selection(
        MapperManager* local_mapper, VariantImpl* impl, Processor::Kind kind,
        const std::deque<InstanceSet>& physical_instances,
        const char* mapper_call_name) const
    //--------------------------------------------------------------------------
    {
      // Check the concurrent constraints
      if (impl->is_concurrent() && !concurrent_task && !must_epoch_task &&
          is_index_space && (index_domain.get_volume() > 1))
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error
            << "Mapper " << *local_mapper << " has mapped " << *this
            << " to a concurrent task variant " << impl->get_name()
            << " but this task was not launched in a concurrent index space "
            << "task launch or must epoch launch. Concurrent task variants can "
            << "only be used in concurrent index space task launches or must "
            << "epoch launches.";
        error.raise();
      }
      else if (
          concurrent_task && !must_epoch_task && !impl->is_concurrent() &&
          is_index_space && (index_domain.get_volume() > 1))
      {
        Warning warning;
        warning
            << "Mapper " << *local_mapper
            << " selected non-concurrent task variant " << impl->get_name()
            << " for " << *this
            << " which was launched as a concurrent index space task launch. "
            << "Concurrent index space task launches have additional overhead "
            << "associated with them so you should really only use them if you "
            << "intend to use concurrent task variants. Also note this warning "
            << "may turn into an error if any of the point tasks of this index "
            << "task selected a concurrent variant.";
        warning.raise();
      }
      // Check the layout constraints first
      const TaskLayoutConstraintSet& layout_constraints =
          impl->get_layout_constraints();
      unsigned req_id = 0;
      unsigned cur_id = 0;
      // fields explicitly specified in any constraint for region requirement
      std::set<FieldID> explicit_fields, align_fields, offset_fields;
      for (const std::pair<const unsigned, LayoutConstraintID>&
               constraint_pair : layout_constraints.layouts)
      {
        // obtain all fields explicitly specified in task layout constraints for
        // a region requirement
        cur_id = constraint_pair.first;
        if (req_id == cur_id)
        {
          explicit_fields.clear();
          align_fields.clear();
          offset_fields.clear();
          req_id++;
          for (std::multimap<unsigned, LayoutConstraintID>::const_iterator
                   lay_it = layout_constraints.layouts.lower_bound(
                       constraint_pair.first);
               lay_it !=
               layout_constraints.layouts.upper_bound(constraint_pair.first);
               lay_it++)
          {
            // Get the layout constraints from the task layout set
            const LayoutConstraints* index_constraints =
                runtime->find_layout_constraints(lay_it->second);
            const std::vector<FieldID>& constraint_fields =
                index_constraints->field_constraint.get_field_set();
            // check if there are any field constraints in the current task
            // layout constraint
            for (FieldID fid : constraint_fields)
            {
              // check if the field is included in the needed_fields
              std::set<FieldID>::const_iterator finder =
                  explicit_fields.find(fid);
              if (finder == explicit_fields.end())
              {
                explicit_fields.insert(fid);
              }
            }
            // alignment constraints may have an explicit field
            if (!index_constraints->alignment_constraints.empty())
            {
              for (unsigned idx = 0;
                   idx < index_constraints->alignment_constraints.size(); idx++)
              {
                FieldID fid = index_constraints->alignment_constraints[idx].fid;
                std::set<FieldID>::const_iterator finder =
                    explicit_fields.find(fid);
                if (finder == explicit_fields.end())
                {
                  explicit_fields.insert(fid);
                  align_fields.insert(fid);
                }
              }
            }
            // offset constraints may have an explicit field
            if (!index_constraints->offset_constraints.empty())
            {
              for (unsigned idx = 0;
                   idx < index_constraints->offset_constraints.size(); idx++)
              {
                FieldID fid = index_constraints->offset_constraints[idx].fid;
                std::set<FieldID>::const_iterator finder =
                    explicit_fields.find(fid);
                if (finder == explicit_fields.end())
                {
                  explicit_fields.insert(fid);
                  align_fields.insert(fid);
                }
              }
            }
          }
        }
        // Might have constraints for extra region requirements
        if (constraint_pair.first >= physical_instances.size())
          continue;
        const InstanceSet& instances =
            physical_instances[constraint_pair.first];
        if (IS_NO_ACCESS(regions[constraint_pair.first]))
          continue;
        LayoutConstraints* constraints =
            runtime->find_layout_constraints(constraint_pair.second);

        const std::vector<FieldID>& field_vec =
            constraints->field_constraint.field_set;
        FieldMask constraint_mask;
        FieldSpaceNode* field_node = runtime->get_node(
            regions[constraint_pair.first].region.get_field_space());
        std::set<FieldID> field_set(field_vec.begin(), field_vec.end());
        if (!field_vec.empty())
        {
          constraint_mask = field_node->get_field_mask(field_set);
        }
        else if (!constraints->alignment_constraints.empty())
        {
          constraint_mask = field_node->get_field_mask(align_fields);
        }
        else if (!constraints->offset_constraints.empty())
        {
          constraint_mask = field_node->get_field_mask(offset_fields);
        }
        else
        {
          // task layout constraint without explicit fields can
          // apply to remaining fields in the region requirement
          constraint_mask = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES) ^
                            field_node->get_field_mask(explicit_fields);
        }
        const LayoutConstraint* conflict_constraint = nullptr;
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          const InstanceRef& ref = instances[idx];
          // Check to see if we have any fields which overlap
          const FieldMask overlap = constraint_mask & ref.get_valid_fields();
          if (!overlap)
            continue;
          InstanceManager* manager = ref.get_manager();
          if (manager->conflicts(constraints, &conflict_constraint))
            break;
          // Check to see if we need an exact match on the layouts
          // Either because it was asked for or because the task
          // variant needs padding and therefore must match precisely
          if (constraints->specialized_constraint.is_exact() ||
              (constraints->padding_constraint.delta.get_dim() > 0))
          {
            std::vector<LogicalRegion> regions_to_check(
                1, regions[constraint_pair.first].region);
            PhysicalManager* phy = manager->as_physical_manager();
            if (!phy->meets_regions(
                    regions_to_check,
                    constraints->specialized_constraint.is_exact(),
                    &constraints->padding_constraint.delta))
            {
              if (constraints->specialized_constraint.is_exact())
                conflict_constraint = &constraints->specialized_constraint;
              else
                conflict_constraint = &constraints->padding_constraint;
              break;
            }
          }
        }
        if (conflict_constraint != nullptr)
        {
          if (local_mapper == nullptr)
            local_mapper = runtime->find_mapper(current_proc, map_id);
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output. Mapper "
                << local_mapper->get_mapper_name() << "selected variant "
                << *impl << " for " << *this
                << ", but instance selected for region requirement "
                << constraint_pair.first
                << " fails to satisfy the corresponding "
                << conflict_constraint->get_constraint_kind()
                << " layout constraint.";
          error.raise();
        }
      }
      // Now we can test against the execution constraints
      const ExecutionConstraintSet& execution_constraints =
          impl->get_execution_constraints();
      // TODO: Check ISA, resource, and launch constraints
      // First check the processor constraint
      if (execution_constraints.processor_constraint.is_valid())
      {
        // If the constraint is a no processor constraint we can ignore it
        if (!execution_constraints.processor_constraint.can_use(kind))
        {
          if (local_mapper == nullptr)
            local_mapper = runtime->find_mapper(current_proc, map_id);
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output. Mapper "
                << local_mapper->get_mapper_name() << " selected variant "
                << *impl << " for " << *this
                << ". However, this variant does not permit running on " << kind
                << " processors.";
          error.raise();
        }
      }
      // Then check the colocation constraints
      for (const ColocationConstraint& constraint :
           execution_constraints.colocation_constraints)
      {
        if (constraint.indexes.size() < 2)
          continue;
        bool first = true;
        DistributedID tree_id = 0;
        FieldSpaceNode* field_space_node = nullptr;
        std::map<
            unsigned /*field index*/, std::pair<PhysicalManager*, unsigned> >
            colocation_instances;
        for (const unsigned& region_idx : constraint.indexes)
        {
          legion_assert(
              regions[region_idx].handle_type == LEGION_SINGULAR_PROJECTION);
          const RegionRequirement& req = regions[region_idx];
          if (first)
          {
            first = false;
            tree_id = req.region.get_tree_id();
            field_space_node = runtime->get_node(req.region.get_field_space());
            const InstanceSet& insts = physical_instances[region_idx];
            FieldMask colocation_mask;
            if (constraint.fields.empty())
            {
              // If there are no explicit fields then we are
              // just going through and checking all of them
              for (const FieldID& field_id : req.privilege_fields)
              {
                unsigned index = field_space_node->get_field_index(field_id);
                colocation_instances[index] =
                    std::pair<PhysicalManager*, unsigned>(nullptr, region_idx);
                colocation_mask.set_bit(index);
              }
            }
            else
            {
              for (const FieldID& field_id : constraint.fields)
              {
                if (req.privilege_fields.find(field_id) ==
                    req.privilege_fields.end())
                  continue;
                unsigned index = field_space_node->get_field_index(field_id);
                colocation_instances[index] =
                    std::pair<PhysicalManager*, unsigned>(nullptr, region_idx);
                colocation_mask.set_bit(index);
              }
            }
            for (unsigned idx = 0; idx < insts.size(); idx++)
            {
              const InstanceRef& ref = insts[idx];
              const FieldMask overlap =
                  colocation_mask & ref.get_valid_fields();
              if (!overlap)
                continue;
              InstanceManager* man = ref.get_manager();
              if (man->is_virtual_manager())
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error << "Invalid mapper output. Mapper " << *local_mapper
                      << " selected a virtual instance for region requirement "
                      << idx << " of " << *this
                      << ", but also selected variant " << *impl
                      << " which contains a colocation constraint for "
                      << "this region requirement. It is illegal to request a "
                      << "virtual mapping for a region requirement with a "
                      << "colocation constraint.";
                error.raise();
              }
              PhysicalManager* manager = man->as_physical_manager();
              int index = overlap.find_first_set();
              while (index >= 0)
              {
                std::map<unsigned, std::pair<PhysicalManager*, unsigned> >::
                    iterator finder = colocation_instances.find(index);
                legion_assert(finder != colocation_instances.end());
                legion_assert(finder->second.first == nullptr);
                legion_assert(finder->second.second == region_idx);
                finder->second.first = manager;
                index = overlap.find_next_set(index + 1);
              }
            }
          }
          else
          {
            // check to make sure that all these region requirements have
            // the same region tree ID.
            if (req.region.get_tree_id() != tree_id)
            {
              Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              error << "Invalid location constraint. Location constraint "
                    << "specified on region requirements "
                    << *(constraint.indexes.begin()) << " and " << region_idx
                    << " of variant " << *impl << " of " << *this
                    << ", but region requirements contain regions that "
                    << "from different region trees (" << tree_id << " and "
                    << req.region.get_tree_id()
                    << "). Colocation constraints must always "
                    << "be specified on region requirements with regions "
                    << "from the same region tree.";
              error.raise();
            }
            const InstanceSet& insts = physical_instances[region_idx];
            if (local_mapper == nullptr)
              local_mapper = runtime->find_mapper(current_proc, map_id);
            for (unsigned idx = 0; idx < insts.size(); idx++)
            {
              const InstanceRef& ref = insts[idx];
              InstanceManager* man = ref.get_manager();
              if (man->is_virtual_manager())
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error
                    << "Invalid mapper output. Mapper "
                    << local_mapper->get_mapper_name()
                    << " selected a virtual instance for region requirement "
                    << region_idx << " of " << *this
                    << ", but also selected variant " << *impl
                    << " which contains a colocation constraint for this "
                    << "region requirement. It is illegal to request a virtual "
                    << "mapping for a region requirement with a colocation "
                       "constraint.";
                error.raise();
              }
              PhysicalManager* manager = man->as_physical_manager();
              const FieldMask& inst_mask = ref.get_valid_fields();
              std::vector<FieldID> field_names;
              field_space_node->get_field_set(
                  inst_mask, parent_ctx, field_names);
              unsigned name_index = 0;
              int index = inst_mask.find_first_set();
              while (index >= 0)
              {
                std::map<unsigned, std::pair<PhysicalManager*, unsigned> >::
                    const_iterator finder = colocation_instances.find(index);
                if (finder != colocation_instances.end())
                {
                  if (finder->second.first->get_instance() !=
                      manager->get_instance())
                  {
                    Error error(LEGION_MAPPER_EXCEPTION);
                    error << "Invalid mapper output. Mapper "
                          << local_mapper->get_mapper_name()
                          << " selected variant " << *impl << " for " << *this
                          << ". However, this variant requires that field "
                          << field_names[name_index]
                          << " of region requirements " << region_idx
                          << " be co-located with prior requirement "
                          << finder->second.second
                          << " but it is not. Requirement " << region_idx
                          << " mapped to instance " << manager->get_instance()
                          << " while prior requirement "
                          << finder->second.second << " mapped to instance "
                          << finder->second.first->get_instance() << ".";
                    error.raise();
                  }
                }
                else
                {
                  if (!constraint.fields.empty())
                  {
                    if (constraint.fields.find(field_names[name_index]) !=
                        constraint.fields.end())
                      colocation_instances[index] =
                          std::make_pair(manager, region_idx);
                  }
                  else
                    colocation_instances[index] =
                        std::make_pair(manager, region_idx);
                }
                index = inst_mask.find_next_set(index + 1);
                name_index++;
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::compute_parent_indexes(bool force)
    //--------------------------------------------------------------------------
    {
      parent_req_indexes.resize(logical_regions.size());
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
      {
        if (!force && runtime->safe_model)
          verify_requirement(logical_regions[idx], idx, is_index_space);
        parent_req_indexes[idx] = parent_ctx->find_parent_region_index(
            this, logical_regions[idx], idx, false /*skip*/, force);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_children_committed(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      if (precondition.exists() && !precondition.has_triggered())
      {
        DeferTriggerChildrenCommitArgs args(this);
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, precondition);
        return;
      }
      bool task_commit = false;
      {
        AutoLock o_lock(op_lock);
        // There is a small race condition here which is alright
        // as long as we haven't committed yet
        legion_assert(!children_commit);
        children_commit = true;
        task_commit = commit_received;
      }
      if (task_commit)
        trigger_task_commit();
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::log_requirement(
        UniqueID uid, unsigned idx, const RegionRequirement& req)
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level == NO_SPY_LOGGING)
        return;
      const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                       (req.handle_type == LEGION_REGION_PROJECTION);
      const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                        (req.handle_type == LEGION_PARTITION_PROJECTION);

      LegionSpy::log_logical_requirement(
          uid, idx, reg,
          reg ? req.region.index_space.get_id() :
                req.partition.index_partition.get_id(),
          reg ? req.region.field_space.get_id() :
                req.partition.field_space.get_id(),
          reg ? req.region.get_tree_id() : req.partition.get_tree_id(),
          req.privilege, req.prop, req.redop, req.parent.index_space.get_id());
      LegionSpy::log_requirement_fields(uid, idx, req.privilege_fields);
      if (proj)
        LegionSpy::log_requirement_projection(uid, idx, req.projection);
    }

    /////////////////////////////////////////////////////////////
    // Task Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    TaskImpl::TaskImpl(TaskID tid, const char* name /*=nullptr*/)
      : task_id(tid),
        initial_name(static_cast<char*>(malloc(
            ((name == nullptr) ? 64 : strlen(name) + 1) * sizeof(char)))),
        all_idempotent(false)
    //--------------------------------------------------------------------------
    {
      // Always fill in semantic info 0 with a name for the task
      if (name != nullptr)
      {
        const size_t name_size = strlen(name) + 1;  // for \0
        semantic_infos[LEGION_NAME_SEMANTIC_TAG] =
            SemanticInfo(name, name_size, false /*mutable*/);
        LegionSpy::log_task_name(task_id, name);
        // Also set the initial name to be safe
        memcpy(initial_name, name, name_size);
        // Register this task with the profiler if necessary
        if (runtime->profiler != nullptr)
          runtime->profiler->register_task_kind(task_id, name, false);
      }
      else  // Just set the initial name
      {
        snprintf(initial_name, 64, "unnamed_task_%d", task_id);
        // Register this task with the profiler if necessary
        if (runtime->profiler != nullptr)
          runtime->profiler->register_task_kind(task_id, initial_name, false);
      }
    }

    //--------------------------------------------------------------------------
    TaskImpl::~TaskImpl(void)
    //-------------------------------------------------------------------------
    {
      semantic_infos.clear();
      free(initial_name);
    }

    //--------------------------------------------------------------------------
    VariantID TaskImpl::get_unique_variant_id(void)
    //--------------------------------------------------------------------------
    {
      AutoLock t_lock(task_lock);
      // VariantIDs have to uniquely identify our node so start at our
      // current runtime name and stride by the number of nodes
      VariantID result = runtime->address_space;
      if (result == 0)  // Never use VariantID 0
        result = runtime->runtime_stride;
      for (; result <=
             (std::numeric_limits<unsigned>::max() - runtime->runtime_stride);
           result += runtime->runtime_stride)
      {
        if (variants.find(result) != variants.end())
          continue;
        if (pending_variants.find(result) != pending_variants.end())
          continue;
        pending_variants.insert(result);
        return result;
      }
      std::abort();
    }

    //--------------------------------------------------------------------------
    void TaskImpl::add_variant(VariantImpl* impl)
    //--------------------------------------------------------------------------
    {
      legion_assert(impl->owner == this);
      AutoLock t_lock(task_lock);
      if (!variants.empty())
      {
        if (all_idempotent != impl->is_idempotent())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error
              << "Variants of task " << get_name(false /*need lock*/) << " (ID "
              << task_id << ") have different idempotent options. "
              << "All variants of the same task must all be either idempotent "
              << "or non-idempotent.";
          error.raise();
        }
      }
      else
        all_idempotent = impl->is_idempotent();
      // Check to see if this variant has already been registered
      if (variants.find(impl->vid) != variants.end())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Duplicate variant ID " << impl->vid << " registered for task "
              << get_name(false /*need lock*/) << " (ID " << task_id << ").";
        error.raise();
      }
      variants[impl->vid] = impl;
      // Erase the pending VariantID if there is one
      pending_variants.erase(impl->vid);
    }

    //--------------------------------------------------------------------------
    VariantImpl* TaskImpl::find_variant_impl(
        VariantID variant_id, bool can_fail)
    //--------------------------------------------------------------------------
    {
      // See if we already have the variant
      {
        AutoLock t_lock(task_lock, false /*exclusive*/);
        std::map<VariantID, VariantImpl*>::const_iterator finder =
            variants.find(variant_id);
        if (finder != variants.end())
          return finder->second;
      }
      if (!can_fail)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Unable to find variant " << variant_id << " of task "
              << get_name() << "!";
        error.raise();
      }
      return nullptr;
    }

    //--------------------------------------------------------------------------
    void TaskImpl::find_valid_variants(
        std::vector<VariantID>& valid_variants, Processor::Kind kind) const
    //--------------------------------------------------------------------------
    {
      if (kind == Processor::NO_KIND)
      {
        AutoLock t_lock(task_lock, false /*exclusive*/);
        valid_variants.resize(variants.size());
        unsigned idx = 0;
        for (const std::pair<const VariantID, VariantImpl*>& variant_pair :
             variants)
        {
          valid_variants[idx++] = variant_pair.first;
        }
      }
      else
      {
        AutoLock t_lock(task_lock, false /*exclusive*/);
        for (const std::pair<const VariantID, VariantImpl*>& variant_pair :
             variants)
        {
          if (variant_pair.second->can_use(kind, true /*warn*/))
            valid_variants.emplace_back(variant_pair.first);
        }
      }
    }

    //--------------------------------------------------------------------------
    const char* TaskImpl::get_name(bool needs_lock /*= true*/)
    //--------------------------------------------------------------------------
    {
      if (needs_lock)
      {
        // Do the request through the semantic information
        const void* ptr = nullptr;
        size_t dummy_size;
        if (retrieve_semantic_information(
                LEGION_NAME_SEMANTIC_TAG, ptr, dummy_size, true /*can fail*/,
                false /*wait until*/))
        {
          const char* result = nullptr;
          static_assert(sizeof(result) == sizeof(ptr));
          memcpy(&result, &ptr, sizeof(result));
          return result;
        }
      }
      else
      {
        // If we're already holding the lock then we can just do
        // the local look-up regardless of if we're the owner or not
        std::map<SemanticTag, SemanticInfo>::const_iterator finder =
            semantic_infos.find(LEGION_NAME_SEMANTIC_TAG);
        if (finder != semantic_infos.end())
          return static_cast<const char*>(finder->second.buffer.get_buffer());
      }
      // Couldn't find it so use the initial name
      return initial_name;
    }

    //--------------------------------------------------------------------------
    void TaskImpl::attach_semantic_information(
        SemanticTag tag, AddressSpaceID source, const void* buffer, size_t size,
        bool is_mutable, bool send_to_owner)
    //--------------------------------------------------------------------------
    {
      if ((tag == LEGION_NAME_SEMANTIC_TAG) && (runtime->profiler != nullptr))
        runtime->profiler->register_task_kind(
            task_id, (const char*)buffer, true);

      bool added = true;
      RtUserEvent to_trigger;
      {
        AutoLock t_lock(task_lock);
        std::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_infos.find(tag);
        if (finder != semantic_infos.end())
        {
          // Check to see if it is valid
          if (finder->second.is_valid())
          {
            // See if it is mutable or not
            if (!finder->second.is_mutable)
            {
              // Note mutable so check to make sure that the bits are the same
              if (size != finder->second.buffer.get_size())
              {
                Error error(LEGION_INTERFACE_EXCEPTION);
                error << "Inconsistent Semantic Tag value "
                      << "for tag " << tag << " with different sizes of "
                      << size << " and " << finder->second.buffer.get_size()
                      << " for task impl";
                error.raise();
              }
              // Otherwise do a bitwise comparison
              {
                const char* orig =
                    (const char*)finder->second.buffer.get_buffer();
                const char* next = (const char*)buffer;
                for (unsigned idx = 0; idx < size; idx++)
                {
                  char diff = orig[idx] ^ next[idx];
                  if (diff)
                  {
                    Error error(LEGION_INTERFACE_EXCEPTION);
                    error << "Inconsistent Semantic Tag value "
                          << "for tag " << tag << " with different values at"
                          << "byte " << idx << " for task impl, " << orig[idx]
                          << " != " << next[idx];
                    error.raise();
                  }
                }
              }
              added = false;
            }
            else
            {
              // It is mutable so just overwrite it
              finder->second.buffer.save_buffer(buffer, size);
              finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
              finder->second.is_mutable = is_mutable;
            }
          }
          else
          {
            finder->second.buffer.save_buffer(buffer, size);
            to_trigger = finder->second.ready_event;
            finder->second.ready_event = RtUserEvent::NO_RT_USER_EVENT;
            finder->second.is_mutable = is_mutable;
          }
        }
        else
          semantic_infos[tag] = SemanticInfo(buffer, size, is_mutable);
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      if (added)
      {
        if (send_to_owner)
        {
          AddressSpaceID owner_space = get_owner_space();
          // if we are not the owner and the message didn't come
          // from the owner, then send it
          if ((owner_space != runtime->address_space) &&
              (source != owner_space))
          {
            if (tag == LEGION_NAME_SEMANTIC_TAG)
            {
              // Special case here for task names, the user can reasonably
              // expect all tasks to have an initial name so we have to
              // guarantee that this update is propagated before continuing
              // because otherwise we can't distinguish the case where a
              // name hasn't propagated from one where it was never set
              RtUserEvent wait_on = Runtime::create_rt_user_event();
              send_semantic_info(
                  owner_space, tag, buffer, size, is_mutable, wait_on);
              wait_on.wait();
            }
            else
              send_semantic_info(owner_space, tag, buffer, size, is_mutable);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool TaskImpl::retrieve_semantic_information(
        SemanticTag tag, const void*& result, size_t& size, bool can_fail,
        bool wait_until)
    //--------------------------------------------------------------------------
    {
      RtEvent wait_on;
      RtUserEvent request;
      const AddressSpaceID owner_space = get_owner_space();
      const bool is_remote = (owner_space != runtime->address_space);
      {
        AutoLock t_lock(task_lock);
        std::map<SemanticTag, SemanticInfo>::const_iterator finder =
            semantic_infos.find(tag);
        if (finder != semantic_infos.end())
        {
          // Already have the data so we are done
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            return true;
          }
          else if (is_remote)
          {
            if (can_fail)
            {
              // Have to make our own event
              request = Runtime::create_rt_user_event();
              wait_on = request;
            }
            else  // can use the canonical event
              wait_on = finder->second.ready_event;
          }
          else if (wait_until)  // local so use the canonical event
            wait_on = finder->second.ready_event;
        }
        else
        {
          // Otherwise we make an event to wait on
          if (!can_fail && wait_until)
          {
            // Make a canonical ready event
            request = Runtime::create_rt_user_event();
            semantic_infos[tag] = SemanticInfo(request);
            wait_on = request;
          }
          else if (is_remote)
          {
            // Make an event just for us to use
            request = Runtime::create_rt_user_event();
            wait_on = request;
          }
        }
      }
      // We didn't find it yet, see if we have something to wait on
      if (!wait_on.exists())
      {
        // Nothing to wait on so we have to do something
        if (can_fail)
          return false;
        Error e(LEGION_INTERFACE_EXCEPTION);
        e << "Invalid semantic tag " << tag << " for task implementation";
        e.raise();
      }
      else
      {
        // Send a request if necessary
        if (is_remote && request.exists())
          send_semantic_request(
              owner_space, tag, can_fail, wait_until, request);
        wait_on.wait();
      }
      // When we wake up, we should be able to find everything
      AutoLock t_lock(task_lock, false /*exclusive*/);
      std::map<SemanticTag, SemanticInfo>::const_iterator finder =
          semantic_infos.find(tag);
      if (finder == semantic_infos.end())
      {
        if (can_fail)
          return false;
        Error e(LEGION_INTERFACE_EXCEPTION);
        e << "invalid semantic tag " << tag << " for task implementation";
        e.raise();
      }
      result = finder->second.buffer.get_buffer();
      size = finder->second.buffer.get_size();
      return true;
    }

    //--------------------------------------------------------------------------
    void TaskImpl::send_semantic_info(
        AddressSpaceID target, SemanticTag tag, const void* buffer, size_t size,
        bool is_mutable, RtUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      TaskSemanticInfoResponse rez;
      {
        RezCheck z(rez);
        rez.serialize(task_id);
        rez.serialize(tag);
        rez.serialize(size);
        rez.serialize(buffer, size);
        rez.serialize(is_mutable);
        rez.serialize(to_trigger);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void TaskImpl::send_semantic_request(
        AddressSpaceID target, SemanticTag tag, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      TaskSemanticInfoRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(task_id);
        rez.serialize(tag);
        rez.serialize(can_fail);
        rez.serialize(wait_until);
        rez.serialize(ready);
      }
      rez.dispatch(target);
    }

    //--------------------------------------------------------------------------
    void TaskImpl::process_semantic_request(
        SemanticTag tag, AddressSpaceID target, bool can_fail, bool wait_until,
        RtUserEvent ready)
    //--------------------------------------------------------------------------
    {
      legion_assert(get_owner_space() == runtime->address_space);
      RtEvent precondition;
      const void* result = nullptr;
      size_t size = 0;
      bool is_mutable = false;
      {
        AutoLock t_lock(task_lock);
        // See if we already have the data
        std::map<SemanticTag, SemanticInfo>::iterator finder =
            semantic_infos.find(tag);
        if (finder != semantic_infos.end())
        {
          if (finder->second.is_valid())
          {
            result = finder->second.buffer.get_buffer();
            size = finder->second.buffer.get_size();
            is_mutable = finder->second.is_mutable;
          }
          else if (!can_fail && wait_until)
            precondition = finder->second.ready_event;
        }
        else if (!can_fail && wait_until)
        {
          // Don't have it yet, make a condition and hope that one comes
          RtUserEvent ready_event = Runtime::create_rt_user_event();
          precondition = ready_event;
          semantic_infos[tag] = SemanticInfo(ready_event);
        }
      }
      if (result == nullptr)
      {
        // this will cause a failure on the original node
        if (can_fail || !wait_until)
          Runtime::trigger_event(ready);
        else
        {
          // Defer this until the semantic condition is ready
          SemanticRequestArgs args(this, tag, target);
          runtime->issue_runtime_meta_task(
              args, LG_LATENCY_WORK_PRIORITY, precondition);
        }
      }
      else
        send_semantic_info(target, tag, result, size, is_mutable, ready);
    }

    //--------------------------------------------------------------------------
    void TaskImpl::SemanticRequestArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->process_semantic_request(
          tag, source, false, false, RtUserEvent::NO_RT_USER_EVENT);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskSemanticInfoRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      TaskID task_id;
      derez.deserialize(task_id);
      SemanticTag tag;
      derez.deserialize(tag);
      bool can_fail;
      derez.deserialize(can_fail);
      bool wait_until;
      derez.deserialize(wait_until);
      RtUserEvent ready;
      derez.deserialize(ready);
      TaskImpl* impl = runtime->find_or_create_task_impl(task_id);
      impl->process_semantic_request(tag, source, can_fail, wait_until, ready);
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskSemanticInfoResponse::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      TaskID task_id;
      derez.deserialize(task_id);
      SemanticTag tag;
      derez.deserialize(tag);
      size_t size;
      derez.deserialize(size);
      const void* buffer = derez.get_current_pointer();
      derez.advance_pointer(size);
      bool is_mutable;
      derez.deserialize(is_mutable);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      TaskImpl* impl = runtime->find_or_create_task_impl(task_id);
      impl->attach_semantic_information(
          tag, source, buffer, size, is_mutable, false /*send to owner*/);
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ AddressSpaceID TaskImpl::get_owner_space(TaskID task_id)
    //--------------------------------------------------------------------------
    {
      return (task_id % runtime->total_address_spaces);
    }

    /////////////////////////////////////////////////////////////
    // Variant Impl
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    VariantImpl::VariantImpl(
        VariantID v, TaskImpl* own, const TaskVariantRegistrar& registrar,
        size_t return_size, bool has_return_size, const CodeDescriptor& realm,
        const void* udata /*=nullptr*/, size_t udata_size /*=0*/)
      : vid(v), owner(own), global(registrar.global_registration),
        needs_padding(check_padding(registrar.layout_constraints)),
        has_return_type_size(has_return_size), return_type_size(return_size),
        descriptor_id(runtime->get_unique_code_descriptor_id()),
        realm_descriptor(realm),
        execution_constraints(registrar.execution_constraints),
        layout_constraints(registrar.layout_constraints),
        leaf_pool_bounds(make_pool_bounds(own, v, registrar)),
        user_data_size(udata_size), leaf_variant(registrar.leaf_variant),
        inner_variant(registrar.inner_variant),
        idempotent_variant(registrar.idempotent_variant),
        replicable_variant(registrar.replicable_variant),
        concurrent_variant(registrar.concurrent_variant),
        concurrent_barrier(registrar.concurrent_barrier)
    //--------------------------------------------------------------------------
    {
      if (udata != nullptr)
      {
        user_data = malloc(user_data_size);
        memcpy(user_data, udata, user_data_size);
      }
      else
        user_data = nullptr;
      // If we have a variant name, then record it
      if (registrar.task_variant_name == nullptr)
      {
        variant_name = (char*)malloc(64 * sizeof(char));
        snprintf(variant_name, 64, "unnamed_variant_%d", vid);
      }
      else
        variant_name = strdup(registrar.task_variant_name);
      // If a global registration was requested, but the code descriptor
      // provided does not have portable implementations, try to make one
      // (if it fails, we'll complain below)
      if (global && !realm_descriptor.has_portable_implementations())
        realm_descriptor.create_portable_implementation();
      // Perform the registration, the normal case is not to have separate
      // runtime instances, but if we do have them, we only register on
      // the local processor
      Realm::ProfilingRequestSet profiling_requests;
      const ProcessorConstraint& proc_constraint =
          execution_constraints.processor_constraint;
      if (proc_constraint.valid_kinds.empty())
      {
        Warning warning;
        warning << "NO PROCESSOR CONSTRAINT SPECIFIED FOR VARIANT"
                << " " << variant_name << " (ID " << vid << ") OF TASK "
                << owner->get_name(false) << " (ID " << owner->task_id << ")!"
                << " ASSUMING LOC_PROC!";
        warning.raise();
        ready_event = ApEvent(Processor::register_task_by_kind(
            Processor::LOC_PROC, false /*global*/, descriptor_id,
            realm_descriptor, profiling_requests, user_data, user_data_size));
      }
      else if (proc_constraint.valid_kinds.size() > 1)
      {
        std::set<ApEvent> ready_events;
        for (const Processor::Kind& proc_kind : proc_constraint.valid_kinds)
          ready_events.insert(ApEvent(Processor::register_task_by_kind(
              proc_kind, false /*global*/, descriptor_id, realm_descriptor,
              profiling_requests, user_data, user_data_size)));
        ready_event = Runtime::merge_events(nullptr, ready_events);
      }
      else
        ready_event = ApEvent(Processor::register_task_by_kind(
            proc_constraint.valid_kinds[0], false /*global*/, descriptor_id,
            realm_descriptor, profiling_requests, user_data, user_data_size));
      // register this with the runtime profiler if we have to
      if (runtime->profiler != nullptr)
        runtime->profiler->register_task_variant(
            own->task_id, vid, variant_name);
      // Check that global registration has portable implementations
      if (global && (!realm_descriptor.has_portable_implementations()))
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Variant " << variant_name
              << " requested global registration without "
              << "a portable implementation.";
        error.raise();
      }
      if (leaf_variant && inner_variant)
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Task variant " << variant_name << " (ID " << vid
              << ") of task " << owner->get_name() << " (ID " << owner->task_id
              << ") is not "
              << "permitted to be both inner and leaf tasks "
              << "simultaneously.";
        error.raise();
      }
      if (runtime->record_registration)
        log_registration.print(
            "Task variant %s of task %s (ID %d) has Realm ID %ld", variant_name,
            owner->get_name(), owner->task_id, descriptor_id);
    }

    //--------------------------------------------------------------------------
    VariantImpl::~VariantImpl(void)
    //--------------------------------------------------------------------------
    {
      if (user_data != nullptr)
        free(user_data);
      if (variant_name != nullptr)
        free(variant_name);
    }

    //--------------------------------------------------------------------------
    bool VariantImpl::is_no_access_region(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      bool result = false;
      for (std::multimap<unsigned, LayoutConstraintID>::const_iterator it =
               layout_constraints.layouts.lower_bound(idx);
           it != layout_constraints.layouts.upper_bound(idx); ++it)
      {
        result = true;
        LayoutConstraints* constraints =
            runtime->find_layout_constraints(it->second);
        if (!constraints->specialized_constraint.is_no_access())
        {
          result = false;
          break;
        }
      }
      return result;
    }

    //--------------------------------------------------------------------------
    ApEvent VariantImpl::dispatch_task(
        Processor target, SingleTask* task, TaskContext* ctx,
        ApEvent precondition, int priority,
        Realm::ProfilingRequestSet& requests)
    //--------------------------------------------------------------------------
    {
      // Either it is local or it is a group that we made
      legion_assert(
          runtime->is_local(target) ||
          (target.kind() == Processor::PROC_GROUP));
      // Add any profiling requests
      if (runtime->profiler != nullptr)
        runtime->profiler->add_task_request(
            requests, owner->task_id, vid, task->get_unique_op_id(), target,
            precondition);
      // Increment the number of outstanding tasks
      runtime->increment_total_outstanding_tasks(task->task_id, false /*meta*/);
      if (ready_event.exists())
        return ApEvent(target.spawn(
            descriptor_id, &ctx, sizeof(ctx), requests,
            Runtime::merge_events(nullptr, precondition, ready_event),
            priority));
      return ApEvent(target.spawn(
          descriptor_id, &ctx, sizeof(ctx), requests, precondition, priority));
    }

    //--------------------------------------------------------------------------
    bool VariantImpl::can_use(Processor::Kind kind, bool warn) const
    //--------------------------------------------------------------------------
    {
      const ProcessorConstraint& constraint =
          execution_constraints.processor_constraint;
      if (constraint.is_valid())
        return constraint.can_use(kind);
      if (warn)
      {
        Warning warning;
        warning << "NO PROCESSOR CONSTRAINT SPECIFIED FOR VARIANT"
                << " " << variant_name << " (ID " << vid << ") OF TASK "
                << owner->get_name(false) << " (ID " << owner->task_id << ")!"
                << " ASSUMING LOC_PROC!";
        warning.raise();
      }
      return (Processor::LOC_PROC == kind);
    }

    //--------------------------------------------------------------------------
    void VariantImpl::broadcast_variant(
        RtUserEvent done, AddressSpaceID origin, AddressSpaceID local)
    //--------------------------------------------------------------------------
    {
      std::vector<AddressSpaceID> targets;
      std::vector<AddressSpaceID> locals;
      const AddressSpaceID start = local * runtime->legion_collective_radix + 1;
      for (int idx = 0; idx < runtime->legion_collective_radix; idx++)
      {
        AddressSpaceID next = start + idx;
        if (next >= runtime->total_address_spaces)
          break;
        locals.emplace_back(next);
        // Convert from relative to actual address space
        AddressSpaceID actual = (origin + next) % runtime->total_address_spaces;
        targets.emplace_back(actual);
      }
      if (!targets.empty())
      {
        std::set<RtEvent> local_done;
        for (unsigned idx = 0; idx < targets.size(); idx++)
        {
          RtUserEvent next_done = Runtime::create_rt_user_event();
          VariantBroadcast rez;
          {
            RezCheck z(rez);
            // pack the code descriptors
            Realm::Serialization::ByteCountSerializer counter;
            realm_descriptor.serialize(counter, true /*portable*/);
            const size_t impl_size = counter.bytes_used();
            rez.serialize(impl_size);
            {
              Realm::Serialization::FixedBufferSerializer serializer(
                  rez.reserve_bytes(impl_size), impl_size);
              realm_descriptor.serialize(serializer, true /*portable*/);
            }
            rez.serialize(owner->task_id);
            rez.serialize(vid);
            // Extra padding to fix a realm bug for now
            rez.serialize(vid);
            rez.serialize(next_done);
            rez.serialize(return_type_size);
            rez.serialize(has_return_type_size);
            rez.serialize(user_data_size);
            if (user_data_size > 0)
              rez.serialize(user_data, user_data_size);
            rez.serialize(leaf_variant);
            if (leaf_variant)
            {
              rez.serialize<size_t>(leaf_pool_bounds.size());
              for (const std::pair<
                       const std::pair<Memory::Kind, bool>, PoolBounds>&
                       pool_pair : leaf_pool_bounds)
              {
                rez.serialize(pool_pair.first.first);
                rez.serialize(pool_pair.first.second);
                rez.serialize(pool_pair.second);
              }
            }
            rez.serialize(inner_variant);
            rez.serialize(idempotent_variant);
            rez.serialize(replicable_variant);
            size_t name_size = strlen(variant_name) + 1;
            rez.serialize(variant_name, name_size);
            // Pack the constraints
            execution_constraints.serialize(rez);
            layout_constraints.serialize(rez);
            rez.serialize(origin);
            rez.serialize(locals[idx]);
          }
          rez.dispatch(targets[idx]);
          local_done.insert(next_done);
        }
        Runtime::trigger_event(done, Runtime::merge_events(local_done));
      }
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void VariantImpl::find_padded_locks(
        SingleTask* task, const std::vector<RegionRequirement>& regions,
        const std::deque<InstanceSet>& physical_instances) const
    //--------------------------------------------------------------------------
    {
      legion_assert(needs_padding);
      for (const std::pair<const unsigned, LayoutConstraintID>&
               constraint_pair : layout_constraints.layouts)
      {
        const LayoutConstraints* layout =
            runtime->find_layout_constraints(constraint_pair.second);
        if (layout->padding_constraint.delta.get_dim() == 0)
          continue;
        legion_assert(constraint_pair.first < regions.size());
        legion_assert(constraint_pair.first < physical_instances.size());
        const RegionRequirement& req = regions[constraint_pair.first];
        const InstanceSet& instances =
            physical_instances[constraint_pair.first];
        // Check to see if we have any explicit fields
        std::set<FieldID> padded_fields;
        if (!layout->field_constraint.field_set.empty())
        {
          for (const FieldID& field_id : layout->field_constraint.field_set)
          {
            legion_assert(
                req.privilege_fields.find(field_id) !=
                req.privilege_fields.end());
            padded_fields.insert(field_id);
          }
        }
        else  // Add all the fields for this region requirement
          padded_fields.insert(
              req.privilege_fields.begin(), req.privilege_fields.end());
        FieldSpaceNode* fs = runtime->get_node(req.region.get_field_space());
        FieldMask padded_mask = fs->get_field_mask(padded_fields);
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          const InstanceRef& ref = instances[idx];
          const FieldMask& overlap = padded_mask & ref.get_valid_fields();
          if (!overlap)
            continue;
          PhysicalManager* manager = ref.get_physical_manager();
          manager->find_padded_reservations(
              overlap, task, constraint_pair.first);
          padded_mask -= overlap;
          if (!padded_mask)
            break;
        }
        legion_assert(!padded_mask);
      }
    }

    //--------------------------------------------------------------------------
    void VariantImpl::record_padded_fields(
        const std::vector<RegionRequirement>& regions,
        const std::vector<PhysicalRegion>& physical_regions) const
    //--------------------------------------------------------------------------
    {
      legion_assert(needs_padding);
      for (const std::pair<const unsigned, LayoutConstraintID>&
               constraint_pair : layout_constraints.layouts)
      {
        const LayoutConstraints* layout =
            runtime->find_layout_constraints(constraint_pair.second);
        if (layout->padding_constraint.delta.get_dim() == 0)
          continue;
        legion_assert(constraint_pair.first < regions.size());
        legion_assert(constraint_pair.first < physical_regions.size());
        const RegionRequirement& req = regions[constraint_pair.first];
        const PhysicalRegion& region = physical_regions[constraint_pair.first];
        // Check to see if we have any explicit fields
        if (layout->field_constraint.field_set.empty())
        {
          // Add all the fields for this region requirement
          for (const FieldID& field_id : req.privilege_fields)
            region.impl->add_padded_field(field_id);
        }
        else
        {
          // Only add the fields specified by the constraint
          for (const FieldID& field_id : layout->field_constraint.field_set)
          {
            legion_assert(
                req.privilege_fields.find(field_id) !=
                req.privilege_fields.end());
            region.impl->add_padded_field(field_id);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ bool VariantImpl::check_padding(
        const TaskLayoutConstraintSet& constraints)
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const unsigned, LayoutConstraintID>&
               constraint_pair : constraints.layouts)
      {
        const LayoutConstraints* layout =
            runtime->find_layout_constraints(constraint_pair.second);
        if (layout->padding_constraint.delta.get_dim() > 0)
          return true;
      }
      return false;
    }

    //--------------------------------------------------------------------------
    /*static*/ std::map<std::pair<Memory::Kind, bool>, PoolBounds>
        VariantImpl::make_pool_bounds(
            TaskImpl* task, VariantID v, const TaskVariantRegistrar& registrar)
    //--------------------------------------------------------------------------
    {
      std::map<std::pair<Memory::Kind, bool>, PoolBounds> pool_bounds;
      for (const std::pair<const Memory::Kind, PoolBounds>& bound :
           registrar.leaf_pool_bounds)
        pool_bounds.emplace(std::make_pair(
            std::make_pair(bound.first, true /*escaping*/), bound.second));
      for (const std::pair<const Memory::Kind, PoolBounds>& bound :
           registrar.non_escaping_leaf_pool_bounds)
      {
        if (!bound.second.is_bounded())
        {
          Warning warning;
          warning << "Detected non-escaping unbounded pool requested in "
                  << bound.first << " memory for variant " << v << " of "
                  << task->get_name() << ". ";
          if (pool_bounds
                  .emplace(std::make_pair(
                      std::make_pair(bound.first, true /*escaping*/),
                      bound.second))
                  .second)
            warning << "Promoted this an escaping memory pool.";
          else
            warning << "Unable to promote this to an escaping memory pool "
                    << "since an escaping pool was already specified for this "
                    << "kind of memory.";
          warning.raise();
        }
        else
          pool_bounds.emplace(std::make_pair(
              std::make_pair(bound.first, false /*escaping*/), bound.second));
      }
      // Hope the compiler does move return-value optimization
      return pool_bounds;
    }

    //--------------------------------------------------------------------------
    /*static*/ void VariantBroadcast::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t impl_size;
      derez.deserialize(impl_size);
      CodeDescriptor realm_desc;
      {
        // Realm's serializers assume properly aligned buffers, so
        // malloc a temporary buffer here and copy the data to ensure
        // alignment.
        void* impl_buffer = malloc(impl_size);
        legion_assert(impl_buffer);
        memcpy(impl_buffer, derez.get_current_pointer(), impl_size);
        derez.advance_pointer(impl_size);
        Realm::Serialization::FixedBufferDeserializer deserializer(
            impl_buffer, impl_size);
        legion_no_skip_assert(realm_desc.deserialize(deserializer));
        free(impl_buffer);
      }
      TaskID task_id;
      derez.deserialize(task_id);
      TaskVariantRegistrar registrar(task_id, false /*global*/);
      VariantID variant_id;
      derez.deserialize(variant_id);
      // Extra padding to fix a realm bug for now
      derez.deserialize(variant_id);
      RtUserEvent done;
      derez.deserialize(done);
      size_t return_type_size;
      derez.deserialize(return_type_size);
      bool has_return_type_size;
      derez.deserialize(has_return_type_size);
      size_t user_data_size;
      derez.deserialize(user_data_size);
      const void* user_data = derez.get_current_pointer();
      derez.advance_pointer(user_data_size);
      derez.deserialize(registrar.leaf_variant);
      if (registrar.leaf_variant)
      {
        size_t num_pools;
        derez.deserialize(num_pools);
        for (unsigned idx = 0; idx < num_pools; idx++)
        {
          Memory::Kind memkind;
          derez.deserialize(memkind);
          bool escaping;
          derez.deserialize(escaping);
          if (escaping)
            derez.deserialize(registrar.leaf_pool_bounds[memkind]);
          else
            derez.deserialize(registrar.non_escaping_leaf_pool_bounds[memkind]);
        }
      }
      derez.deserialize(registrar.inner_variant);
      derez.deserialize(registrar.idempotent_variant);
      derez.deserialize(registrar.replicable_variant);
      // The last thing will be the name
      registrar.task_variant_name = (const char*)derez.get_current_pointer();
      size_t name_size = strlen(registrar.task_variant_name) + 1;
      derez.advance_pointer(name_size);
      // Unpack the constraints
      registrar.execution_constraints.deserialize(derez);
      registrar.layout_constraints.deserialize(derez);
      // Ask the runtime to perform the registration
      // Can lie about preregistration since the user would already have
      // gotten there error message on the owner node
      runtime->register_variant(
          registrar, user_data, user_data_size, realm_desc, return_type_size,
          has_return_type_size, variant_id, false /*check task*/,
          false /*check context*/, true /*preregistered*/);
      AddressSpaceID origin;
      derez.deserialize(origin);
      AddressSpaceID local;
      derez.deserialize(local);
      VariantImpl* impl = runtime->find_variant_impl(task_id, variant_id);
      impl->broadcast_variant(done, origin, local);
    }

  }  // namespace Internal
}  // namespace Legion
