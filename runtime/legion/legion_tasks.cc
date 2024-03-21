/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#include "legion/region_tree.h"
#include "legion/legion_tasks.h"
#include "legion/legion_spy.h"
#include "legion/legion_trace.h"
#include "legion/legion_context.h"
#include "legion/legion_profiling.h"
#include "legion/legion_instances.h"
#include "legion/legion_analysis.h"
#include "legion/legion_views.h"
#include "legion/legion_replication.h"

#include <algorithm>

#define PRINT_REG(reg) (reg).index_space.id,(reg).field_space.id, (reg).tree_id

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // External Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ExternalTask::ExternalTask(void)
      : Task(), arg_manager(NULL)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ExternalTask::pack_external_task(Serializer &rez,
                                          AddressSpaceID target) const
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      rez.serialize(task_id);
      rez.serialize(indexes.size());
      for (unsigned idx = 0; idx < indexes.size(); idx++)
        pack_index_space_requirement(indexes[idx], rez);
      rez.serialize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        pack_region_requirement(regions[idx], rez);
      rez.serialize(output_regions.size());
      for (unsigned idx = 0; idx < output_regions.size(); idx++)
        pack_output_requirement(output_regions[idx], rez);
      rez.serialize(futures.size());
      // If we are remote we can just do the normal pack
      for (std::vector<Future>::const_iterator it =
            futures.begin(); it != futures.end(); it++)
        if (it->impl != NULL)
          it->impl->pack_future(rez, target);
        else
          rez.serialize<DistributedID>(0);
      rez.serialize(grants.size());
      for (unsigned idx = 0; idx < grants.size(); idx++)
        pack_grant(grants[idx], rez);
      rez.serialize(wait_barriers.size());
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        pack_phase_barrier(wait_barriers[idx], rez);
      rez.serialize(arrive_barriers.size());
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        pack_phase_barrier(arrive_barriers[idx], rez);
      rez.serialize<bool>((arg_manager != NULL));
      rez.serialize(arglen);
      rez.serialize(args,arglen);
      pack_mappable(*this, rez);
      rez.serialize(is_index_space);
      rez.serialize(concurrent_task);
      rez.serialize(must_epoch_task);
      rez.serialize(index_domain);
      rez.serialize(index_point);
      rez.serialize(sharding_space);
      rez.serialize(local_arglen);
      rez.serialize(local_args,local_arglen);
      rez.serialize(orig_proc);
      // No need to pack current proc, it will get set when we unpack
      rez.serialize(steal_count);
      // No need to pack remote, it will get set
      rez.serialize(speculated);
      // No need to pack local function, it's not if we're sending this remote
      rez.serialize<uint64_t>(get_context_index());
    }

    //--------------------------------------------------------------------------
    void ExternalTask::unpack_external_task(Deserializer &derez,
                                            Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      derez.deserialize(task_id);
      size_t num_indexes;
      derez.deserialize(num_indexes);
      indexes.resize(num_indexes);
      for (unsigned idx = 0; idx < indexes.size(); idx++)
        unpack_index_space_requirement(indexes[idx], derez);
      size_t num_regions;
      derez.deserialize(num_regions);
      regions.resize(num_regions);
      for (unsigned idx = 0; idx < regions.size(); idx++)
        unpack_region_requirement(regions[idx], derez); 
      size_t num_output_regions;
      derez.deserialize(num_output_regions);
      output_regions.resize(num_output_regions);
      for (unsigned idx = 0; idx < output_regions.size(); idx++)
        unpack_output_requirement(output_regions[idx], derez);
      size_t num_futures;
      derez.deserialize(num_futures);
      futures.resize(num_futures);
      for (unsigned idx = 0; idx < futures.size(); idx++)
        futures[idx] = FutureImpl::unpack_future(runtime, derez);
      size_t num_grants;
      derez.deserialize(num_grants);
      grants.resize(num_grants);
      for (unsigned idx = 0; idx < grants.size(); idx++)
        unpack_grant(grants[idx], derez);
      size_t num_wait_barriers;
      derez.deserialize(num_wait_barriers);
      wait_barriers.resize(num_wait_barriers);
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
        unpack_phase_barrier(wait_barriers[idx], derez);
      size_t num_arrive_barriers;
      derez.deserialize(num_arrive_barriers);
      arrive_barriers.resize(num_arrive_barriers);
      for (unsigned idx = 0; idx < arrive_barriers.size(); idx++)
        unpack_phase_barrier(arrive_barriers[idx], derez);
      bool has_arg_manager;
      derez.deserialize(has_arg_manager);
      derez.deserialize(arglen);
      if (arglen > 0)
      {
        if (has_arg_manager)
        {
#ifdef DEBUG_LEGION
          assert(arg_manager == NULL);
#endif
          arg_manager = new AllocManager(arglen);
          arg_manager->add_reference();
          args = arg_manager->get_allocation();
        }
        else
          args = legion_malloc(TASK_ARGS_ALLOC, arglen);
        derez.deserialize(args,arglen);
      }
      unpack_mappable(*this, derez); 
      derez.deserialize(is_index_space);
      derez.deserialize(concurrent_task);
      derez.deserialize(must_epoch_task);
      derez.deserialize(index_domain);
      derez.deserialize(index_point);
      derez.deserialize(sharding_space);
      derez.deserialize(local_arglen);
      if (local_arglen > 0)
      {
        local_args = malloc(local_arglen);
        derez.deserialize(local_args,local_arglen);
      }
      derez.deserialize(orig_proc);
      derez.deserialize(steal_count);
      derez.deserialize(speculated);
      uint64_t index;
      derez.deserialize(index);
      set_context_index(index);
    } 

    //--------------------------------------------------------------------------
    /*static*/ void ExternalTask::pack_output_requirement(
                                  const OutputRequirement &req, Serializer &rez)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_region_requirement(req, rez);
      rez.serialize(req.type_tag);
      rez.serialize(req.field_space);
      rez.serialize(req.global_indexing);
      rez.serialize(req.valid_requirement);
      rez.serialize(req.color_space);
    }

    //--------------------------------------------------------------------------
    /*static*/ void ExternalTask::unpack_output_requirement(
                                    OutputRequirement &req, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_region_requirement(req, derez);
      derez.deserialize(req.type_tag);
      derez.deserialize(req.field_space);
      derez.deserialize(req.global_indexing);
      derez.deserialize(req.valid_requirement);
      derez.deserialize(req.color_space);
    }

    /////////////////////////////////////////////////////////////
    // Task Operation 
    /////////////////////////////////////////////////////////////
  
    //--------------------------------------------------------------------------
    TaskOp::TaskOp(Runtime *rt)
      : ExternalTask(), PredicatedOp(rt), 
        logical_regions(TaskRequirements(*this))
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    TaskOp::~TaskOp(void)
    //--------------------------------------------------------------------------
    {
    }

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
#ifdef DEBUG_LEGION
      assert(parent_ctx != NULL);
#endif
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
      if (parent_task == NULL)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const char* TaskOp::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      TaskImpl *impl = runtime->find_or_create_task_impl(task_id);
      return impl->get_name();
    }

    //--------------------------------------------------------------------------
    bool TaskOp::is_reducing_future(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_remote_operation(Serializer &rez, AddressSpaceID target,
                                       std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_local_remote_operation(rez);
      pack_external_task(rez, target);
      pack_profiling_requests(rez, applied_events);
    }
    
    //--------------------------------------------------------------------------
    void TaskOp::pack_profiling_requests(Serializer &rez,
                                         std::set<RtEvent> &applied) const
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
    void TaskOp::set_current_proc(Processor current)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(current.exists());
      assert(runtime->is_local(current));
#endif
      // Always clear target_proc and the mapper when setting a new current proc
      mapper = NULL;
      current_proc = current;
      target_proc = current;
    }

    //--------------------------------------------------------------------------
    void TaskOp::activate(void)
    //--------------------------------------------------------------------------
    {
      PredicatedOp::activate();
      complete_received = false;
      commit_received = false;
      children_complete = false;
      children_commit = false;
      stealable = false;
      options_selected = false;
      map_origin = false;
      request_valid_instances = false;
      elide_future_return = false;
      replicate = false; 
      local_cached = false;
      arg_manager = NULL;
      target_proc = Processor::NO_PROC;
      mapper = NULL;
      must_epoch = NULL;
      must_epoch_task = false;
      concurrent_task = false;
      local_function = false;
      orig_proc = Processor::NO_PROC; // for is_remote
    }

    //--------------------------------------------------------------------------
    void TaskOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      PredicatedOp::deactivate(freeop);
      indexes.clear();
      regions.clear();
      output_regions.clear();
      futures.clear();
      grants.clear();
      wait_barriers.clear();
      arrive_barriers.clear();
      if (args != NULL)
      {
        if (arg_manager != NULL)
        {
          // If the arg manager is not NULL then we delete the
          // argument manager and just zero out the arguments
          if (arg_manager->remove_reference())
            delete (arg_manager);
          arg_manager = NULL;
        }
        else
          legion_free(TASK_ARGS_ALLOC, args, arglen);
        args = NULL;
        arglen = 0;
      }
      if (local_args != NULL)
      {
        free(local_args);
        local_args = NULL;
        local_arglen = 0;
      }
      if (mapper_data != NULL)
      {
        free(mapper_data);
        mapper_data = NULL;
        mapper_data_size = 0;
      }
      check_collective_regions.clear();
      atomic_locks.clear(); 
      parent_req_indexes.clear();
      version_infos.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
    }

    //--------------------------------------------------------------------------
    void TaskOp::set_must_epoch(MustEpochOp *epoch, unsigned index,
                                bool do_registration)
    //--------------------------------------------------------------------------
    {
      Operation::set_must_epoch(epoch, do_registration);
      must_epoch_index = index;
      must_epoch_task = true;
      concurrent_task = false;
      if (runtime->legion_spy_enabled)
      {
        const TaskKind kind = get_task_kind();
        if (kind == INDEX_TASK_KIND)
          LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                    unique_op_id, task_id, get_task_name());
        else if (kind == INDIVIDUAL_TASK_KIND)
          LegionSpy::log_individual_task(parent_ctx->get_unique_id(),
                                    unique_op_id, task_id, get_task_name());
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::pack_base_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_BASE_TASK_CALL);
      // pack all the user facing data first
      pack_external_task(rez, target); 
      RezCheck z(rez);
      rez.serialize(parent_req_indexes.size());
      for (unsigned idx = 0; idx < parent_req_indexes.size(); idx++)
        rez.serialize(parent_req_indexes[idx]);
      rez.serialize(map_origin);
      if (map_origin)
      {
        rez.serialize<size_t>(atomic_locks.size());
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      else
      {
        rez.serialize(memo_state);
        if (memo_state == MEMO_RECORD)
        {
          rez.serialize(tpl);
          rez.serialize(trace_local_id);
        }
        rez.serialize<size_t>(check_collective_regions.size());
        for (unsigned idx = 0; idx < check_collective_regions.size(); idx++)
          rez.serialize(check_collective_regions[idx]);
      }
      rez.serialize(request_valid_instances);
      rez.serialize(execution_fence_event);
      rez.serialize(elide_future_return);
      rez.serialize(replicate);
      rez.serialize(true_guard);
      rez.serialize(false_guard);
    }

    //--------------------------------------------------------------------------
    void TaskOp::unpack_base_task(Deserializer &derez,
                                  std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, UNPACK_BASE_TASK_CALL);
      // unpack all the user facing data
      unpack_external_task(derez, runtime); 
      DerezCheck z(derez);
      size_t num_indexes;
      derez.deserialize(num_indexes);
      if (num_indexes > 0)
      {
        parent_req_indexes.resize(num_indexes);
        for (unsigned idx = 0; idx < num_indexes; idx++)
          derez.deserialize(parent_req_indexes[idx]);
      }
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
        derez.deserialize(memo_state);
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
      derez.deserialize(request_valid_instances);
      derez.deserialize(execution_fence_event);
      derez.deserialize(elide_future_return);
      derez.deserialize(replicate);
      derez.deserialize(true_guard);
      derez.deserialize(false_guard);
      // Already had our options selected
      options_selected = true;
    }

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::process_unpack_task(Runtime *rt,Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Figure out what kind of task this is and where it came from
      DerezCheck z(derez);
      Processor current;
      derez.deserialize(current);
      TaskKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case INDIVIDUAL_TASK_KIND:
          {
            IndividualTask *task = rt->get_available_individual_task();
            std::set<RtEvent> ready_events;
            if (task->unpack_task(derez, current, ready_events))
            {
              RtEvent ready;
              if (!ready_events.empty())
                ready = Runtime::merge_events(ready_events);
              // Origin mapped tasks can go straight to launching 
              // themselves since they are already mapped
              if (task->is_origin_mapped())
              {
                TriggerTaskArgs trigger_args(task);
                rt->issue_runtime_meta_task(trigger_args, 
                      LG_THROUGHPUT_WORK_PRIORITY, ready);
              }
              else
                task->enqueue_ready_task(false/*target*/, ready);
            }
            break;
          }
        case SLICE_TASK_KIND:
          {
            SliceTask *task = rt->get_available_slice_task();
            std::set<RtEvent> ready_events;
            if (task->unpack_task(derez, current, ready_events))
            {
              RtEvent ready;
              if (!ready_events.empty())
                ready = Runtime::merge_events(ready_events);
              // Origin mapped tasks can go straight to launching 
              // themselves since they are already mapped
              if (task->is_origin_mapped())
              {
                TriggerTaskArgs trigger_args(task);
                rt->issue_runtime_meta_task(trigger_args, 
                      LG_THROUGHPUT_WORK_PRIORITY, ready);
              }
              else
                task->enqueue_ready_task(false/*target*/, ready);
            }
            break;
          }
        case POINT_TASK_KIND:
        case INDEX_TASK_KIND:
        default:
          assert(false); // no other tasks should be sent anywhere
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::process_remote_replay(Runtime *rt, Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      // Figure out what kind of task this is and where it came from
      DerezCheck z(derez);
      ApEvent instance_ready, completion_postcondition;
      derez.deserialize(instance_ready);
      derez.deserialize(completion_postcondition);
      Processor target_proc;
      derez.deserialize(target_proc);
      TaskKind kind;
      derez.deserialize(kind);
      switch (kind)
      {
        case INDIVIDUAL_TASK_KIND:
          {
            IndividualTask *task = rt->get_available_individual_task();
            std::set<RtEvent> ready_events;
            task->unpack_task(derez, target_proc, ready_events);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            task->complete_replay(instance_ready, completion_postcondition);
            break;
          }
        case SLICE_TASK_KIND:
          {
            SliceTask *task = rt->get_available_slice_task();
            std::set<RtEvent> ready_events;
            task->unpack_task(derez, target_proc, ready_events);
            if (!ready_events.empty())
            {
              const RtEvent wait_on = Runtime::merge_events(ready_events);
              if (wait_on.exists() && !wait_on.has_triggered())
                wait_on.wait();
            }
            task->complete_replay(instance_ready, completion_postcondition);
            break;
          }
        case POINT_TASK_KIND:
        case INDEX_TASK_KIND:
        default:
          assert(false); // no other tasks should be sent anywhere
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::mark_stolen(void)
    //--------------------------------------------------------------------------
    {
      steal_count++;
    }

    //--------------------------------------------------------------------------
    void TaskOp::initialize_base_task(InnerContext *ctx,
                const Predicate &p, Processor::TaskFuncID tid, Provenance *prov)
    //--------------------------------------------------------------------------
    {
      initialize_predication(ctx, get_region_count(), p, prov);
      parent_task = ctx->get_task(); // initialize the parent task
      // Fill in default values for all of the Task fields
      orig_proc = ctx->get_executing_processor();
      current_proc = orig_proc;
      steal_count = 0;
      speculated = false;
      local_function = false;
    }

    //--------------------------------------------------------------------------
    void TaskOp::validate_region_requirements(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
      {
        const RegionRequirement &req = logical_regions[idx];
        if (req.privilege != LEGION_NO_ACCESS && req.privilege_fields.empty())
          REPORT_LEGION_WARNING(LEGION_WARNING_REGION_REQUIREMENT_TASK,
                           "REGION REQUIREMENT %d OF "
                           "TASK %s (ID %lld) HAS NO PRIVILEGE "
                           "FIELDS! DID YOU FORGET THEM?!?",
                           idx, get_task_name(), get_unique_id())
        if (IS_READ_ONLY(req) && ((req.privilege & LEGION_DISCARD_INPUT_MASK)
              == LEGION_DISCARD_INPUT_MASK))
          REPORT_LEGION_ERROR(ERROR_INVALID_DISCARD_QUALIFIER,
            "Region requirement %d of %s (UID %lld) combined input-discard "
            "qualifier with read-only privilege which will result in "
            "undefined behavior, therefore this privilege combination is "
            "disallowed.", idx, get_task_name(), get_unique_id())
        if (IS_WRITE_ONLY(req) && ((req.privilege & LEGION_DISCARD_OUTPUT_MASK)
              == LEGION_DISCARD_OUTPUT_MASK))
          REPORT_LEGION_ERROR(ERROR_INVALID_DISCARD_QUALIFIER,
            "Region requirement %d of %s (UID %lld) combined output-discard "
            "qualifier with write-only privilege which will result in "
            "undefined behavior, therefore this privilege combination is "
            "disallowed.", idx, get_task_name(), get_unique_id())
        if (IS_REDUCE(req) && ((req.privilege & (LEGION_DISCARD_INPUT_MASK | 
              LEGION_DISCARD_OUTPUT_MASK)) != LEGION_NO_ACCESS))
          REPORT_LEGION_ERROR(ERROR_INVALID_DISCARD_QUALIFIER,
            "Region requirement %d of %s (UID %lld) combined a discard "
            "qualifier with reduction privilege which will result in "
            "undefined behavior, therefore this privilege combination is "
            "disallowed.", idx, get_task_name(), get_unique_id())
      }
    }

    //--------------------------------------------------------------------------
    bool TaskOp::select_task_options(bool prioritize)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!options_selected);
#endif
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      Mapper::TaskOptions options;
      options.initial_proc = current_proc;
      options.inline_task = false;
      options.stealable = false;
      options.map_locally = false;
      options.valid_instances = mapper->request_valid_instances;
      options.memoize = false;
      options.replicate = false;
      const TaskPriority parent_priority = parent_ctx->is_priority_mutable() ?
        parent_ctx->get_current_priority() : 0;
      options.parent_priority = parent_priority;
      mapper->invoke_select_task_options(this, options, prioritize);
      options_selected = true;
      if (options.initial_proc.kind() == Processor::UTIL_PROC)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
            "Invalid mapper output. Mapper %s requested that task %s (UID %lld)"
            " initially be assigned to a utility processor in "
            "'select_task_options.' Only application processor kinds are "
            "permitted to be the target processor for tasks.",
            mapper->get_mapper_name(), get_task_name(), get_unique_id())
      target_proc = options.initial_proc;
      if (local_function && !runtime->is_local(target_proc))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
            "Invalid mapper output. Mapper %s requested that local function "
            "task %s (UID %lld) be assigned to processor " IDFMT " which is "
            "not local to address space %d. Local function tasks must be "
            "assigned to local processors.", mapper->get_mapper_name(),
            get_task_name(), get_unique_id(),
            target_proc.id, runtime->address_space)
      stealable = options.stealable;
      map_origin = options.map_locally; 
      request_valid_instances = options.valid_instances;
      if (parent_priority != options.parent_priority)
      {
        // Request for priority change see if it is legal or not
        if (parent_ctx->is_priority_mutable())
          parent_ctx->set_current_priority(options.parent_priority);
        else
          REPORT_LEGION_WARNING(LEGION_WARNING_INVALID_PRIORITY_CHANGE,
                                "Mapper %s requested change of priority "
                                "for parent task %s (UID %lld) when launching "
                                "child task %s (UID %lld), but the parent "
                                "context does not support parent task priority "
                                "mutation", mapper->get_mapper_name(),
                                parent_ctx->get_task_name(),
                                parent_ctx->get_unique_id(), 
                                get_task_name(), get_unique_id())
      }
      if (!options.check_collective_regions.empty() && is_index_space)
      {
        for (std::set<unsigned>::const_iterator it =
              options.check_collective_regions.begin(); it !=
              options.check_collective_regions.end(); it++)
        {
          if ((*it) >= regions.size())
            continue;
          const RegionRequirement &req = regions[*it];
          if (IS_NO_ACCESS(req) || req.privilege_fields.empty())
            continue;
          if (IS_WRITE(req))
            REPORT_LEGION_WARNING(LEGION_WARNING_WRITE_PRIVILEGE_COLLECTIVE,
                "Ignoring request by mapper %s to check for collective usage "
                "for region requirement %d of task %s (UID %lld) because "
                "region requirement has writing privileges.",
                mapper->get_mapper_name(), *it, 
                get_task_name(), unique_op_id)
          else
            check_collective_regions.push_back(*it);
        }
        if (!check_collective_regions.empty())
        {
          // Check to make sure that there are no invertible projection functors
          // in this index space launch on writing requirements which might
          // cause point tasks to be interfering. If there are then we can't
          // perform any collective rendezvous here so the tasks map together
          for (unsigned idx = 0; idx < regions.size(); idx++)
          {
            const RegionRequirement &req = regions[idx];
            if (!IS_WRITE(req))
              continue;
            if (((req.projection == 0) &&
                (req.handle_type == LEGION_REGION_PROJECTION)) ||
                runtime->find_projection_function(
                  req.projection)->is_invertible)
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
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
              "Mapper %s request to replicate task %s (UID %lld) that is a "
              "concurrent index space task launch in 'select_task_options'. "
              "It is illegal to replicate the point tasks of a concurrent "
              "index space task launch.", mapper->get_mapper_name(),
              get_task_name(), get_unique_id())
        // Replication of must epoch tasks are not allowed
        if (must_epoch_task)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
              "Mapper %s requested to replicate must epoch task %s (UID %lld). "
              "Replication of must epoch tasks are not supported.",
              mapper->get_mapper_name(), get_task_name(), get_unique_id())
        // Replication of origin-mapped tasks is not supported
        if (map_origin)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_UNSUPPORTED_REPLICATION,
              "Mapper %s requested to both replicate and origin map task %s "
              "(UID %lld) in 'select_task_options'. Replication of origin-"
              "mapped tasks is not currently supported and the request to "
              "replicate the task will be ignored.", mapper->get_mapper_name(),
              get_task_name(), get_unique_id())
          options.replicate = false;
        }
        // Output regions are not currently supported
        if (!output_regions.empty())
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_UNSUPPORTED_REPLICATION,
              "Mapper %s requested to replicate task %s (UID %lld) with output "
              "regions in 'select_task_options'. Legion does not currently "
              "support replication of tasks with output regions at the moment. "
              "You can request support for this feature by emailing the "
              "the Legion developers list or opening a github issue. The "
              "mapper call to replicate_task is being elided.",
              mapper->get_mapper_name(), get_task_name(), get_unique_id())
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
          REPORT_LEGION_WARNING(LEGION_WARNING_UNSUPPORTED_REPLICATION,
              "Mapper %s requested to replicate task %s (UID %lld) with "
              "reduction privilege on region requirement %d in "
              "'select_task_options'. Legion does not currently support "
              "replication of tasks with reduction privileges. "
              "You can request support for this feature by emailing the "
              "Legion developers list or opening a github issue. The mapper "
              "call to replicate_task is being elided.",
              mapper->get_mapper_name(), get_task_name(), get_unique_id(), idx)
          options.replicate = false;
          break;
        }
        replicate = options.replicate;
      }
      if (options.inline_task)
      {
        if (concurrent_task)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
              "Mapper %s requested to inline concurrent task %s (UID %lld). "
              "Inlining of concurrent tasks are not supported.",
              mapper->get_mapper_name(), get_task_name(), get_unique_id())
        if (must_epoch_task)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
              "Mapper %s requested to inline must epoch task %s (UID %lld). "
              "Inlining of must epoch tasks are not supported.",
              mapper->get_mapper_name(), get_task_name(), get_unique_id())
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
    Operation::OpKind TaskOp::get_operation_kind(void) const
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
    void TaskOp::trigger_complete(void) 
    //--------------------------------------------------------------------------
    {
      bool task_complete = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!complete_received);
        assert(!commit_received);
#endif
        complete_received = true;
        // If all our children are also complete then we are done
        task_complete = children_complete;
      }
      if (task_complete)
        trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool task_commit = false; 
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(complete_received);
        assert(!commit_received);
#endif
        commit_received = true;
        // If we already received the child commit then we
        // are ready to commit this task
        task_commit = children_commit;
      }
      if (task_commit)
        trigger_task_commit();
    } 

    //--------------------------------------------------------------------------
    void TaskOp::select_sources(const unsigned index, PhysicalManager *target,
                                const std::vector<InstanceView*> &sources,
                                std::vector<unsigned> &ranking,
                                std::map<unsigned,PhysicalManager*> &points)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < regions.size());
#endif
      Mapper::SelectTaskSrcInput input;
      Mapper::SelectTaskSrcOutput output;
      prepare_for_mapping(target, input.target);
      prepare_for_mapping(sources, input.source_instances,
                          input.collective_views);
      input.region_req_index = index;
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_select_task_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_atomic_locks(const unsigned index,
                                     Reservation lock, bool exclusive)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::map<Reservation,bool>::iterator finder = atomic_locks.find(lock);
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
#ifdef DEBUG_LEGION
      assert(idx < parent_req_indexes.size());
#endif
      return parent_req_indexes[idx];
    }

    //--------------------------------------------------------------------------
    VersionInfo& TaskOp::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    const VersionInfo& TaskOp::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < version_infos.size());
#endif
      return version_infos[idx];
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*,unsigned>* 
                                        TaskOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void TaskOp::defer_distribute_task(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      parent_ctx->add_to_distribute_task_queue(this, precondition);
    }

    //--------------------------------------------------------------------------
    RtEvent TaskOp::defer_perform_mapping(RtEvent precondition, MustEpochOp *op,
                                          const DeferMappingArgs *defer_args,
                                          unsigned invocation_count,
                                          std::vector<unsigned> *performed,
                                          std::vector<ApEvent> *effects)
    //--------------------------------------------------------------------------
    {
      const RtUserEvent done_event = (defer_args == NULL) ? 
        Runtime::create_rt_user_event() : defer_args->done_event;
      DeferMappingArgs args(this, op, done_event, invocation_count,
                            performed, effects);
      runtime->issue_runtime_meta_task(args,
          LG_THROUGHPUT_DEFERRED_PRIORITY, precondition);
      return done_event;
    }

    //--------------------------------------------------------------------------
    void TaskOp::defer_launch_task(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      parent_ctx->add_to_launch_task_queue(this, precondition);
    }

    //--------------------------------------------------------------------------
    void TaskOp::enqueue_ready_task(bool use_target_processor,
                                    RtEvent wait_on /*=RtEvent::NO_RT_EVENT*/)
    //--------------------------------------------------------------------------
    {
      if (use_target_processor)
        set_current_proc(target_proc);
      if (!wait_on.exists() || wait_on.has_triggered())
        runtime->add_to_ready_queue(current_proc, this);
      else
        parent_ctx->add_to_task_queue(this, wait_on);
    }

    //--------------------------------------------------------------------------
    const std::string& TaskOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance *provenance = get_provenance();
      if (provenance != NULL)
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
    void TaskOp::perform_privilege_checks(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, TASK_PRIVILEGE_CHECK_CALL);
      // First check the index privileges
      for (unsigned idx = 0; idx < indexes.size(); idx++)
      {
        LegionErrorType et = parent_ctx->check_privilege(indexes[idx]);
        switch (et)
        {
          case LEGION_NO_ERROR:
            break;
          case ERROR_BAD_PARENT_INDEX:
            {
              REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                              "Parent task %s (ID %lld) of task %s "
                              "(ID %lld) "
                              "does not have an index requirement for "
                              "index space %x as a parent of "
                              "child task's index requirement index %d",
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id(), get_task_name(),
                              get_unique_id(), indexes[idx].parent.id, idx)
              break;
            }
          case ERROR_BAD_INDEX_PATH:
            {
              REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_NOTSUBSPACE,
                              "Index space %x is not a sub-space "
                              "of parent index space %x for index "
                              "requirement %d of task %s (ID %lld)",
                              indexes[idx].handle.id,
                              indexes[idx].parent.id, idx,
                              get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_INDEX_PRIVILEGES:
            {
              REPORT_LEGION_ERROR(ERROR_PRIVILEGES_INDEX_SPACE,
                              "Privileges %x for index space %x "
                              " are not a subset of privileges of parent "
                              "task's privileges for index space "
                              "requirement %d of task %s (ID %lld)",
                              indexes[idx].privilege,
                              indexes[idx].handle.id, idx,
                              get_task_name(), get_unique_id())
              break;
            }
          default:
            assert(false); // Should never happen
        }
      }
      // Now check the region requirement privileges
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Verify that the requirement is self-consistent
        FieldID bad_field = LEGION_AUTO_GENERATE_ID;
        int bad_index = -1;
        LegionErrorType et = runtime->verify_requirement(regions[idx], 
                                                         bad_field); 
        if ((et == LEGION_NO_ERROR) && !is_index_space && 
            ((regions[idx].handle_type == LEGION_PARTITION_PROJECTION) || 
             (regions[idx].handle_type == LEGION_REGION_PROJECTION)))
          et = ERROR_BAD_PROJECTION_USE;
        // If that worked, then check the privileges with the parent context
        if (et == LEGION_NO_ERROR)
          et = parent_ctx->check_privilege(regions[idx], bad_field, bad_index);
        switch (et)
        {
          case LEGION_NO_ERROR:
            break;
          case ERROR_INVALID_REGION_HANDLE:
            {
              REPORT_LEGION_ERROR(ERROR_INVALID_REGION_HANDLE,
                               "Invalid region handle (%x,%d,%d)"
                               " for region requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].region.index_space.id,
                               regions[idx].region.field_space.id,
                               regions[idx].region.tree_id, idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_INVALID_PARTITION_HANDLE:
            {
              REPORT_LEGION_ERROR(ERROR_INVALID_PARTITION_HANDLE,
                               "Invalid partition handle (%x,%d,%d) "
                               "for partition requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].partition.index_partition.id,
                               regions[idx].partition.field_space.id,
                               regions[idx].partition.tree_id, idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_PROJECTION_USE:
            {
              REPORT_LEGION_ERROR(ERROR_PROJECTION_REGION_REQUIREMENT,
                               "Projection region requirement %d used "
                               "in non-index space task %s",
                               idx, get_task_name())
              break;
            }
          case ERROR_NON_DISJOINT_PARTITION:
            {
              REPORT_LEGION_ERROR(ERROR_NONDISJOINT_PARTITION_SELECTED,
                               "Non disjoint partition selected for "
                               "writing region requirement %d of task "
                               "%s.  All projection partitions "
                               "which are not read-only and not reduce "
                               "must be disjoint",
                               idx, get_task_name())
              break;
            }
          case ERROR_FIELD_SPACE_FIELD_MISMATCH:
            {
              FieldSpace sp = 
                (regions[idx].handle_type == LEGION_SINGULAR_PROJECTION) ||
                (regions[idx].handle_type == LEGION_REGION_PROJECTION) ? 
                  regions[idx].region.field_space :
                  regions[idx].partition.field_space;
              REPORT_LEGION_ERROR(ERROR_FIELD_NOT_VALID,
                               "Field %d is not a valid field of field "
                               "space %d for region %d of task %s "
                               "(ID %lld)",
                               bad_field, sp.id, idx, get_task_name(),
                               get_unique_id())
              break;
            }
          case ERROR_INVALID_INSTANCE_FIELD:
            {
              REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_PRIVILEGE,
                               "Instance field %d is not one of the "
                               "privilege fields for region %d of "
                               "task %s (ID %lld)",
                               bad_field, idx, get_task_name(),
                               get_unique_id())
              break;
            }
          case ERROR_DUPLICATE_INSTANCE_FIELD:
            {
              REPORT_LEGION_ERROR(ERROR_INSTANCE_FIELD_DUPLICATE,
                               "Instance field %d is a duplicate for "
                               "region %d of task %s (ID %lld)",
                               bad_field, idx, get_task_name(),
                               get_unique_id())
              break;
            }
          case ERROR_BAD_PARENT_REGION:
            {
              if (bad_index < 0) 
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                                 "Parent task %s (ID %lld) of task %s "
                                 "(ID %lld) does not have a region "
                                 "requirement for region "
                                 "(%x,%x,%x) as a parent of child task's "
                                 "region requirement index %d because "
                                 "no 'parent' region had that name.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 get_task_name(), get_unique_id(),
                                 regions[idx].parent.index_space.id,
                                 regions[idx].parent.field_space.id,
                                 regions[idx].parent.tree_id, idx)
              else if (bad_field == LEGION_AUTO_GENERATE_ID) 
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                                 "Parent task %s (ID %lld) of task %s "
                                 "(ID %lld) does not have a region "
                                 "requirement for region "
                                 "(%x,%x,%x) as a parent of child task's "
                                 "region requirement index %d because "
                                 "parent requirement %d did not have "
                                 "sufficient privileges.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 get_task_name(), get_unique_id(),
                                 regions[idx].parent.index_space.id,
                                 regions[idx].parent.field_space.id,
                                 regions[idx].parent.tree_id, idx, bad_index)
              else 
                REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                                 "Parent task %s (ID %lld) of task %s "
                                 "(ID %lld) does not have a region "
                                 "requirement for region "
                                 "(%x,%x,%x) as a parent of child task's "
                                 "region requirement index %d because "
                                 "parent requirement %d was missing field %d.",
                                 parent_ctx->get_task_name(),
                                 parent_ctx->get_unique_id(),
                                 get_task_name(), get_unique_id(),
                                 regions[idx].parent.index_space.id,
                                 regions[idx].parent.field_space.id,
                                 regions[idx].parent.tree_id, idx,
                                 bad_index, bad_field)
              break;
            }
          case ERROR_BAD_REGION_PATH:
            {
              REPORT_LEGION_ERROR(ERROR_REGION_NOT_SUBREGION,
                               "Region (%x,%x,%x) is not a "
                               "sub-region of parent region "
                               "(%x,%x,%x) for region requirement %d of "
                               "task %s (ID %lld)",
                               regions[idx].region.index_space.id,
                               regions[idx].region.field_space.id,
                               regions[idx].region.tree_id,
                               PRINT_REG(regions[idx].parent), idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_PARTITION_PATH:
            {
              REPORT_LEGION_ERROR(ERROR_PARTITION_NOT_SUBPARTITION,
                               "Partition (%x,%x,%x) is not a "
                               "sub-partition of parent region "
                               "(%x,%x,%x) for region "
                               "requirement %d task %s (ID %lld)",
                               regions[idx].partition.index_partition.id,
                               regions[idx].partition.field_space.id,
                               regions[idx].partition.tree_id,
                               PRINT_REG(regions[idx].parent), idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_REGION_TYPE:
            {
              REPORT_LEGION_ERROR(ERROR_REGION_REQUIREMENT_TASK,
                               "Region requirement %d of task %s "
                               "(ID %lld) "
                               "cannot find privileges for field %d in "
                               "parent task",
                               idx, get_task_name(),
                               get_unique_id(), bad_field)
              break;
            }
          case ERROR_BAD_REGION_PRIVILEGES:
            {
              REPORT_LEGION_ERROR(ERROR_PRIVILEGES_REGION_NOTSUBSET,
                               "Privileges %x for region "
                               "(%x,%x,%x) are not a subset of privileges "
                               "of parent task's privileges for "
                               "region requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].privilege,
                               regions[idx].region.index_space.id,
                               regions[idx].region.field_space.id,
                               regions[idx].region.tree_id, idx,
                               get_task_name(), get_unique_id())
              break;
            }
          case ERROR_BAD_PARTITION_PRIVILEGES:
            {
              REPORT_LEGION_ERROR(ERROR_PRIVILEGES_PARTITION_NOTSUBSET,
                               "Privileges %x for partition (%x,%x,%x) "
                               "are not a subset of privileges of parent "
                               "task's privileges for "
                               "region requirement %d of task %s "
                               "(ID %lld)",
                               regions[idx].privilege,
                               regions[idx].partition.index_partition.id,
                               regions[idx].partition.field_space.id,
                               regions[idx].partition.tree_id, idx,
                               get_task_name(), get_unique_id())
              break;
            }
          default:
            assert(false); // Should never happen
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::clone_task_op_from(TaskOp *rhs, Processor p, 
                                    bool can_steal, bool duplicate_args)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CLONE_TASK_CALL);
#ifdef DEBUG_LEGION
      assert(p.exists());
#endif
      // From Operation
      this->parent_ctx = rhs->parent_ctx;
      this->context_index = rhs->get_context_index();
      this->execution_fence_event = rhs->get_execution_fence_event();
      // Don't register this an operation when setting the must epoch info
      if (rhs->must_epoch != NULL)
        this->set_must_epoch(rhs->must_epoch, rhs->must_epoch_index,
                             false/*do registration*/);
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
      this->arglen = rhs->arglen;
      if (rhs->arg_manager != NULL)
      {
        if (duplicate_args)
        {
#ifdef DEBUG_LEGION
          assert(arg_manager == NULL);
#endif
          this->arg_manager = new AllocManager(this->arglen); 
          this->arg_manager->add_reference();
          this->args = this->arg_manager->get_allocation();
          memcpy(this->args, rhs->args, this->arglen);
        }
        else
        {
          // No need to actually do the copy in this case
          this->arg_manager = rhs->arg_manager; 
          this->arg_manager->add_reference();
          this->args = arg_manager->get_allocation();
        }
      }
      else if (arglen > 0)
      {
        // If there is no argument manager then we do the copy no matter what
        this->args = legion_malloc(TASK_ARGS_ALLOC, arglen);
        memcpy(args,rhs->args,arglen);
      }
      this->map_id = rhs->map_id;
      this->tag = rhs->tag;
      if (rhs->mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(rhs->mapper_data != NULL);
#endif
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
    void TaskOp::update_grants(const std::vector<Grant> &requested_grants)
    //--------------------------------------------------------------------------
    {
      if (requested_grants.empty())
        return;
      grants = requested_grants;
      const ApEvent grant_pre = get_completion_event();
      for (unsigned idx = 0; idx < grants.size(); idx++)
        grants[idx].impl->register_operation(grant_pre);
    }

    //--------------------------------------------------------------------------
    void TaskOp::update_arrival_barriers(
                                const std::vector<PhaseBarrier> &phase_barriers)
    //--------------------------------------------------------------------------
    {
      if (phase_barriers.empty())
        return;
      const ApEvent arrive_pre = get_completion_event();
      for (std::vector<PhaseBarrier>::const_iterator it = 
            phase_barriers.begin(); it != phase_barriers.end(); it++)
      {
        arrive_barriers.push_back(*it);
        Runtime::phase_barrier_arrive(*it, 1/*count*/, arrive_pre);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_phase_barrier_arrival(unique_op_id, it->phase_barrier);
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::compute_point_region_requirements(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, COMPUTE_POINT_REQUIREMENTS_CALL);
      // Update the region requirements for this point
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (regions[idx].handle_type != LEGION_SINGULAR_PROJECTION)
        {
          ProjectionFunction *function = 
            runtime->find_projection_function(regions[idx].projection);
          if (function->is_invertible)
            assert(false); // TODO: implement dependent launches for inline
          regions[idx].region = function->project_point(this, idx, runtime, 
                                                index_domain, index_point);
          // Update the region requirement kind 
          regions[idx].handle_type = LEGION_SINGULAR_PROJECTION;
        }
        // Check to see if the region is a NO_REGION,
        // if it is then switch the privilege to NO_ACCESS
        if (regions[idx].region == LogicalRegion::NO_REGION)
        {
          regions[idx].privilege = LEGION_NO_ACCESS;
          continue;
        }
      }
      complete_point_projection(); 
    }

    //--------------------------------------------------------------------------
    void TaskOp::complete_point_projection(void)
    //--------------------------------------------------------------------------
    {
      SingleTask *single_task = dynamic_cast<SingleTask*>(this);
      if (single_task != NULL)
        single_task->update_no_access_regions();
      // Log our requirements that we computed
      if (runtime->legion_spy_enabled)
      {
        UniqueID our_uid = get_unique_id();
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
          log_requirement(our_uid, idx, logical_regions[idx]);
      }
#ifdef DEBUG_LEGION
      {
        perform_intra_task_alias_analysis();
      }
#endif
    } 

    //--------------------------------------------------------------------------
    bool TaskOp::prepare_steal(void)
    //--------------------------------------------------------------------------
    {
      if (is_origin_mapped())
        return false;
      return true;
    }

    //--------------------------------------------------------------------------
    void TaskOp::finalize_output_region_trees(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!output_regions.empty());
#endif
      const size_t offset = regions.size();
      for (unsigned idx = 0; idx < output_regions.size(); idx++)
        if (!is_output_valid(idx))
          parent_ctx->finalize_output_eqkd_tree(find_parent_index(offset+idx));
    }

    //--------------------------------------------------------------------------
    void TaskOp::perform_intra_task_alias_analysis(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INTRA_TASK_ALIASING_CALL);
      std::map<RegionTreeID,std::vector<unsigned> > tree_indexes;
      // Find the indexes of requirements with the same tree
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
      {
        if (IS_NO_ACCESS(logical_regions[idx]))
          continue;
        tree_indexes[logical_regions[idx].parent.get_tree_id()].push_back(idx);
      }
      // Iterate over the trees with multiple requirements
      for (std::map<RegionTreeID,std::vector<unsigned> >::const_iterator 
            tree_it = tree_indexes.begin(); 
            tree_it != tree_indexes.end(); tree_it++)
      {
        const std::vector<unsigned> &indexes = tree_it->second;
        if (indexes.size() <= 1)
          continue;
        // Get the field masks for each of the requirements
        LegionVector<FieldMask> field_masks(indexes.size());
        std::vector<IndexTreeNode*> index_nodes(indexes.size());
        {
          FieldSpaceNode *field_space_node = 
           runtime->forest->get_node(
               logical_regions[indexes[0]].parent)->column_source;
          for (unsigned idx = 0; idx < indexes.size(); idx++)
          {
            field_masks[idx] = field_space_node->get_field_mask(
                logical_regions[indexes[idx]].privilege_fields);
            if (logical_regions[indexes[idx]].handle_type == 
                LEGION_PARTITION_PROJECTION)
              index_nodes[idx] = runtime->forest->get_node(
                logical_regions[indexes[idx]].partition.get_index_partition());
            else
              index_nodes[idx] = runtime->forest->get_node(
                logical_regions[indexes[idx]].region.get_index_space());
          }
        }
        // Find the sets of fields which are interfering
        for (unsigned i = 1; i < indexes.size(); i++)
        {
          RegionUsage usage1(logical_regions[indexes[i]]);
          for (unsigned j = 0; j < i; j++)
          {
            FieldMask overlap = field_masks[i] & field_masks[j];
            // No field overlap, so there is nothing to do
            if (!overlap)
              continue;
            // No check for region overlap
            IndexTreeNode *common_ancestor = NULL;
            if (runtime->forest->are_disjoint_tree_only(index_nodes[i],
                  index_nodes[j], common_ancestor))
              continue;
#ifdef DEBUG_LEGION
            assert(common_ancestor != NULL); // should have a counterexample
#endif
            // Get the interference kind and report it if it is bad
            RegionUsage usage2(logical_regions[indexes[j]]);
            DependenceType dtype = check_dependence_type<false>(usage1, usage2);
            // We can only reporting interfering requirements precisely
            // if at least one of these is not a projection requireemnts
            // There is a special case here for concurrent tasks with both
            // read-only or reduction requirements, those can still lead to 
            // hangs so we'll report them as interfering
            if ((dtype == LEGION_TRUE_DEPENDENCE) || 
                (dtype == LEGION_ANTI_DEPENDENCE) ||
                (concurrent_task && IS_ATOMIC(usage1) && (usage1 == usage2)))
              report_interfering_requirements(indexes[j], indexes[i]);
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::validate_variant_selection(MapperManager *local_mapper,
                              VariantImpl *impl, Processor::Kind kind, 
                              const std::deque<InstanceSet> &physical_instances,
                              const char *mapper_call_name) const
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, VALIDATE_VARIANT_SELECTION_CALL);
      // Check the concurrent constraints
      if (impl->is_concurrent() && !concurrent_task && !must_epoch_task &&
          is_index_space && (index_domain.get_volume() > 1))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT, "Mapper %s has mapped "
              "task %s (UID %lld) to a concurrent task variant %s but this "
              "task was not launched in a concurrent index space task launch "
              "or must epoch launch. Concurrent task variants can only be used "
              "in concurrent index space task launches or must epoch launches.",
              local_mapper->get_mapper_name(),
              get_task_name(), get_unique_id(), impl->get_name())
      else if (concurrent_task && !impl->is_concurrent() && is_index_space &&
          (index_domain.get_volume() > 1))
        REPORT_LEGION_WARNING(LEGION_WARNING_UNUSED_CONCURRENCY,
            "Mapper %s selected non-concurrent task variant %s for "
            "task %s (UID %lld) which was launched as a concurrent index "
            "space task launch. Concurrent index space task launches have "
            "additional overhead associated with them so you should really "
            "only use them if you intend to use concurrent task variants. "
            "Also note this warning may turn into an error if any of the "
            "point tasks of this index task selected a concurrent variant.",
            local_mapper->get_mapper_name(), impl->get_name(),
            get_task_name(), get_unique_id())
      // Check the layout constraints first
      const TaskLayoutConstraintSet &layout_constraints = 
        impl->get_layout_constraints();
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it = 
            layout_constraints.layouts.begin(); it != 
            layout_constraints.layouts.end(); it++)
      {
        // Might have constraints for extra region requirements
        if (it->first >= physical_instances.size())
          continue;
        const InstanceSet &instances = physical_instances[it->first]; 
        if (IS_NO_ACCESS(regions[it->first]))
          continue;
        LayoutConstraints *constraints = 
          runtime->find_layout_constraints(it->second);
        // If we don't have any fields then this constraint isn't
        // going to apply to any actual instances
        const std::vector<FieldID> &field_vec = 
          constraints->field_constraint.field_set;
        FieldMask constraint_mask;
        if (!field_vec.empty())
        {
          FieldSpaceNode *field_node = runtime->forest->get_node(
                              regions[it->first].region.get_field_space());
          std::set<FieldID> field_set(field_vec.begin(), field_vec.end());
          constraint_mask = field_node->get_field_mask(field_set);
        }
        else
          constraint_mask = FieldMask(LEGION_FIELD_MASK_FIELD_ALL_ONES);
        const LayoutConstraint *conflict_constraint = NULL;
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          const InstanceRef &ref = instances[idx];
          // Check to see if we have any fields which overlap
          const FieldMask overlap = constraint_mask & ref.get_valid_fields();
          if (!overlap)
            continue;
          InstanceManager *manager = ref.get_manager();
          if (manager->conflicts(constraints, &conflict_constraint))
            break;
          // Check to see if we need an exact match on the layouts
          // Either because it was asked for or because the task 
          // variant needs padding and therefore must match precisely
          if (constraints->specialized_constraint.is_exact() ||
              (constraints->padding_constraint.delta.get_dim() > 0))
          {
            std::vector<LogicalRegion> regions_to_check(1, 
                                regions[it->first].region);
            PhysicalManager *phy = manager->as_physical_manager();
            if (!phy->meets_regions(regions_to_check,
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
        if (conflict_constraint != NULL)
        {
          if (local_mapper == NULL)
            local_mapper = runtime->find_mapper(current_proc, map_id);
          const char *constraint_names[] = {
#define CONSTRAINT_NAMES(name, desc) desc,
            LEGION_LAYOUT_CONSTRAINT_KINDS(CONSTRAINT_NAMES)
#undef CONSTRAINT_NAMES
          };
          const char *constraint_name = 
            constraint_names[conflict_constraint->get_constraint_kind()];
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output. Mapper %s selected variant "
                        "%d for task %s (ID %lld). But instance selected "
                        "for region requirement %d fails to satisfy the "
                        "corresponding %s layout constraint.", 
                        local_mapper->get_mapper_name(), impl->vid,
                        get_task_name(), get_unique_id(), it->first,
                        constraint_name)
        }
      }
      // Now we can test against the execution constraints
      const ExecutionConstraintSet &execution_constraints = 
        impl->get_execution_constraints();
      // TODO: Check ISA, resource, and launch constraints
      // First check the processor constraint
      if (execution_constraints.processor_constraint.is_valid())
      {
        // If the constraint is a no processor constraint we can ignore it
        if (!execution_constraints.processor_constraint.can_use(kind))
        {
          if (local_mapper == NULL)
            local_mapper = runtime->find_mapper(current_proc, map_id);
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output. Mapper %s selected variant %d "
                      "for task %s (ID %lld). However, this variant does not "
                      "permit running on processors of kind %s.",
                      local_mapper->get_mapper_name(),
                      impl->vid, get_task_name(), get_unique_id(),
                      Processor::get_kind_name(kind))
        }
      }
      // Then check the colocation constraints
      for (std::vector<ColocationConstraint>::const_iterator con_it = 
            execution_constraints.colocation_constraints.begin(); con_it !=
            execution_constraints.colocation_constraints.end(); con_it++)
      {
        if (con_it->indexes.size() < 2)
          continue;
        unsigned idx = 0;
        bool first = true;
        RegionTreeID tree_id = 0;
        FieldSpaceNode *field_space_node = NULL;
        std::map<unsigned/*field index*/,
          std::pair<PhysicalManager*,unsigned> > colocation_instances;
        for (std::set<unsigned>::const_iterator iit = con_it->indexes.begin();
              iit != con_it->indexes.end(); iit++, idx++)
        {
#ifdef DEBUG_LEGION
          assert(regions[*iit].handle_type == LEGION_SINGULAR_PROJECTION);
#endif
          const RegionRequirement &req = regions[*iit];
          if (first)
          {
            first = false;
            tree_id = req.region.get_tree_id();
            field_space_node = runtime->forest->get_node(
                                req.region.get_field_space());
            const InstanceSet &insts = physical_instances[*iit];
            FieldMask colocation_mask;
            if (con_it->fields.empty())
            {
              // If there are no explicit fields then we are
              // just going through and checking all of them
              for (std::set<FieldID>::const_iterator it = 
                    req.privilege_fields.begin(); it != 
                    req.privilege_fields.end(); it++)
              {
                unsigned index = field_space_node->get_field_index(*it);
                colocation_instances[index] = 
                  std::pair<PhysicalManager*,unsigned>(NULL, *iit);
                colocation_mask.set_bit(index);
              }
            }
            else
            {
              for (std::set<FieldID>::const_iterator it = 
                    con_it->fields.begin(); it != con_it->fields.end(); it++)
              {
                if (req.privilege_fields.find(*it) == 
                    req.privilege_fields.end())
                  continue;
                unsigned index = field_space_node->get_field_index(*it);
                colocation_instances[index] = 
                  std::pair<PhysicalManager*,unsigned>(NULL, *iit);
                colocation_mask.set_bit(index);
              }
            }
            for (unsigned idx = 0; idx < insts.size(); idx++)
            {
              const InstanceRef &ref = insts[idx];
              const FieldMask overlap = 
                colocation_mask & ref.get_valid_fields();
              if (!overlap)
                continue;
              InstanceManager *man = ref.get_manager();
              if (man->is_virtual_manager())
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                    "Invalid mapper output. Mapper %s selected a virtual "
                    "instance for region requirement %d of task %s (UID %lld), "
                    "but also selected variant %d which contains a colocation "
                    "constraint for this region requirement. It is illegal to "
                    "request a virtual mapping for a region requirement with a "
                    "colocation constraint.", local_mapper->get_mapper_name(),
                    *iit, get_task_name(), get_unique_id(), impl->vid)
              PhysicalManager *manager = man->as_physical_manager();
              int index = overlap.find_first_set();
              while (index >= 0)
              {
                std::map<unsigned,
                  std::pair<PhysicalManager*,unsigned> >::iterator finder = 
                    colocation_instances.find(index);
#ifdef DEBUG_LEGION
                assert(finder != colocation_instances.end());
                assert(finder->second.first == NULL);
                assert(finder->second.second == *iit);
#endif
                finder->second.first = manager;
                index = overlap.find_next_set(index+1);
              }
            }
          }
          else
          {
            // check to make sure that all these region requirements have
            // the same region tree ID.
            if (req.region.get_tree_id() != tree_id)
              REPORT_LEGION_ERROR(ERROR_INVALID_LOCATION_CONSTRAINT,
                            "Invalid location constraint. Location constraint "
                            "specified on region requirements %d and %d of "
                            "variant %d of task %s, but region requirements "
                            "contain regions that from different region trees "
                            "(%d and %d). Colocation constraints must always "
                            "be specified on region requirements with regions "
                            "from the same region tree.", 
                            *(con_it->indexes.begin()), *iit, impl->vid,
                            get_task_name(), tree_id, 
                            req.region.get_tree_id())
            const InstanceSet &insts = physical_instances[*iit];
            if (local_mapper == NULL)
              local_mapper = runtime->find_mapper(current_proc, map_id);
            for (unsigned idx = 0; idx < insts.size(); idx++)
            {
              const InstanceRef &ref = insts[idx];
              InstanceManager *man = ref.get_manager();
              if (man->is_virtual_manager())
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                    "Invalid mapper output. Mapper %s selected a virtual "
                    "instance for region requirement %d of task %s (UID %lld), "
                    "but also selected variant %d which contains a colocation "
                    "constraint for this region requirement. It is illegal to "
                    "request a virtual mapping for a region requirement with a "
                    "colocation constraint.", local_mapper->get_mapper_name(),
                    *iit, get_task_name(), get_unique_id(), impl->vid)
              PhysicalManager *manager = man->as_physical_manager();
              const FieldMask &inst_mask = ref.get_valid_fields();
              std::vector<FieldID> field_names;
              field_space_node->get_field_set(inst_mask,parent_ctx,field_names);
              unsigned name_index = 0;
              int index = inst_mask.find_first_set();
              while (index >= 0)
              {
                std::map<unsigned,
                  std::pair<PhysicalManager*,unsigned> >::const_iterator
                    finder = colocation_instances.find(index);
                if (finder != colocation_instances.end())
                {
                  if (finder->second.first->get_instance() != 
                      manager->get_instance())
                    REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output. Mapper %s selected variant "
                          "%d for task %s (ID %lld). However, this variant "
                          "requires that field %d of region requirements %d be "
                          "co-located with prior requirement %d but it is not. "
                          "Requirement %d mapped to instance " IDFMT " while "
                          "prior requirement %d mapped to instance " IDFMT "",
                          local_mapper->get_mapper_name(), impl->vid, 
                          get_task_name(), get_unique_id(), 
                          field_names[name_index], *iit, finder->second.second,
                          *iit, manager->get_instance().id, 
                          finder->second.second,
                          finder->second.first->get_instance().id)
                }
                else
                {
                  if (!con_it->fields.empty())
                  {
                    if (con_it->fields.find(field_names[name_index]) !=
                        con_it->fields.end())
                      colocation_instances[index] = 
                        std::make_pair(manager, *iit);
                  }
                  else
                    colocation_instances[index] = std::make_pair(manager, *iit);
                }
                index = inst_mask.find_next_set(index+1);
                name_index++;
              }
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::compute_parent_indexes(InnerContext *alt_context/*= NULL*/)
    //--------------------------------------------------------------------------
    {
      parent_req_indexes.resize(get_region_count());
      InnerContext *use_ctx = (alt_context == NULL) ? parent_ctx : alt_context;
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
      {
        int parent_index = 
          use_ctx->find_parent_region_req(logical_regions[idx]);
        if (parent_index < 0)
          REPORT_LEGION_ERROR(ERROR_PARENT_TASK_TASK,
                           "Parent task %s (ID %lld) of task %s "
                           "(ID %lld) does not have a region "
                           "requirement for region "
                           "(%x,%x,%x) as a parent of child task's "
                           "region requirement index %d",
                           use_ctx->get_task_name(), 
                           use_ctx->get_unique_id(),
                           get_task_name(), get_unique_id(),
                           logical_regions[idx].parent.index_space.id,
                           logical_regions[idx].parent.field_space.id, 
                           logical_regions[idx].parent.tree_id, idx)
        parent_req_indexes[idx] = parent_index;
      }
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_children_complete(void)
    //--------------------------------------------------------------------------
    {
      bool task_complete = false;
      {
        AutoLock o_lock(op_lock); 
#ifdef DEBUG_LEGION
        assert(!children_complete);
        // Small race condition here which is alright as
        // long as we haven't committed yet
        assert(!children_commit || !commit_received);
#endif
        children_complete = true;
        task_complete = complete_received;
      }
      if (task_complete)
        trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    void TaskOp::trigger_children_committed(void)
    //--------------------------------------------------------------------------
    {
      bool task_commit = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        // There is a small race condition here which is alright
        // as long as we haven't committed yet
        assert(children_complete || !commit_received);
        assert(!children_commit);
#endif
        children_commit = true;
        task_commit = commit_received;
      }
      if (task_commit)
        trigger_task_commit();
    } 

    //--------------------------------------------------------------------------
    /*static*/ void TaskOp::log_requirement(UniqueID uid, unsigned idx,
                                            const RegionRequirement &req)
    //--------------------------------------------------------------------------
    {
      const bool reg = (req.handle_type == LEGION_SINGULAR_PROJECTION) ||
                       (req.handle_type == LEGION_REGION_PROJECTION);
      const bool proj = (req.handle_type == LEGION_REGION_PROJECTION) ||
                        (req.handle_type == LEGION_PARTITION_PROJECTION); 

      LegionSpy::log_logical_requirement(uid, idx, reg,
          reg ? req.region.index_space.id :
                req.partition.index_partition.id,
          reg ? req.region.field_space.id :
                req.partition.field_space.id,
          reg ? req.region.tree_id : 
                req.partition.tree_id,
          req.privilege, req.prop, req.redop, req.parent.index_space.id);
      LegionSpy::log_requirement_fields(uid, idx, req.privilege_fields);
      if (proj)
        LegionSpy::log_requirement_projection(uid, idx, req.projection);
    }

    ///////////////////////////////////////////////////////////// 
    // Remote Task Op 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    RemoteTaskOp::RemoteTaskOp(Runtime *rt, Operation *ptr, AddressSpaceID src)
      : ExternalTask(), RemoteOp(rt, ptr, src)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    RemoteTaskOp::RemoteTaskOp(const RemoteTaskOp &rhs)
      : ExternalTask(), RemoteOp(rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    RemoteTaskOp::~RemoteTaskOp(void)
    //--------------------------------------------------------------------------
    {
      if (args != NULL)
      {
        if (arg_manager != NULL)
        {
          // If the arg manager is not NULL then we delete the
          // argument manager and just zero out the arguments
          if (arg_manager->remove_reference())
            delete (arg_manager);
          arg_manager = NULL;
        }
        else
          legion_free(TASK_ARGS_ALLOC, args, arglen);
      }
      if (local_args != NULL)
        free(local_args);
    }

    //--------------------------------------------------------------------------
    RemoteTaskOp& RemoteTaskOp::operator=(const RemoteTaskOp &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    UniqueID RemoteTaskOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t RemoteTaskOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::set_context_index(uint64_t index)
    //--------------------------------------------------------------------------
    {
      context_index = index;
    }

    //--------------------------------------------------------------------------
    int RemoteTaskOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    bool RemoteTaskOp::has_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return (get_depth() > 0);
    }

    //--------------------------------------------------------------------------
    const Task* RemoteTaskOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      return parent_ctx->get_task();
    }

    //--------------------------------------------------------------------------
    const std::string& RemoteTaskOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance *provenance = get_provenance();
      if (provenance != NULL)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    const char* RemoteTaskOp::get_task_name(void) const
    //--------------------------------------------------------------------------
    {
      TaskImpl *impl = runtime->find_or_create_task_impl(task_id);
      return impl->get_name();
    }

    //--------------------------------------------------------------------------
    Domain RemoteTaskOp::get_slice_domain(void) const
    //--------------------------------------------------------------------------
    {
      // We're mapping a point task if we've made one of these
      return Domain(index_point, index_point);
    }

    //--------------------------------------------------------------------------
    ShardID RemoteTaskOp::get_shard_id(void) const
    //--------------------------------------------------------------------------
    {
      // We're mapping a point task if we've made one of these
      return 0;
    }

    //--------------------------------------------------------------------------
    size_t RemoteTaskOp::get_total_shards(void) const
    //--------------------------------------------------------------------------
    {
      // We're mapping a point task if we've made one of these
      return 1;
    }

    //--------------------------------------------------------------------------
    DomainPoint RemoteTaskOp::get_shard_point(void) const
    //--------------------------------------------------------------------------
    {
      return DomainPoint(0);
    }

    //--------------------------------------------------------------------------
    Domain RemoteTaskOp::get_shard_domain(void) const
    //--------------------------------------------------------------------------
    {
      return Domain(DomainPoint(0),DomainPoint(0));
    }

    //--------------------------------------------------------------------------
    const char* RemoteTaskOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[TASK_OP_KIND];
    }

    //--------------------------------------------------------------------------
    Operation::OpKind RemoteTaskOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return TASK_OP_KIND;
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::select_sources(const unsigned index,
                                    PhysicalManager *target,
                                    const std::vector<InstanceView*> &sources,
                                    std::vector<unsigned> &ranking,
                                    std::map<unsigned,PhysicalManager*> &points)
    //--------------------------------------------------------------------------
    {
      if (source == runtime->address_space)
      {
        // If we're on the owner node we can just do this
        remote_ptr->select_sources(index, target, sources, ranking, points);
        return;
      }
      Mapper::SelectTaskSrcInput input;
      Mapper::SelectTaskSrcOutput output;
      prepare_for_mapping(sources, input.source_instances,
                          input.collective_views);
      prepare_for_mapping(target, input.target);
      input.region_req_index = index;
      if (mapper == NULL)
        mapper = runtime->find_mapper(map_id);
      mapper->invoke_select_task_sources(this, input, output);
      compute_ranking(mapper, output.chosen_ranking, sources, ranking, points);
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::pack_remote_operation(Serializer &rez,
                 AddressSpaceID target, std::set<RtEvent> &applied_events) const
    //--------------------------------------------------------------------------
    {
      pack_remote_base(rez);
      pack_external_task(rez, target);
      pack_profiling_requests(rez, applied_events);
    }

    //--------------------------------------------------------------------------
    void RemoteTaskOp::unpack(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      unpack_external_task(derez, runtime);
      unpack_profiling_requests(derez);
    }

    /////////////////////////////////////////////////////////////
    // Single Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SingleTask::SingleTask(Runtime *rt)
      : TaskOp(rt)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    SingleTask::~SingleTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void SingleTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, ACTIVATE_SINGLE_CALL);
      TaskOp::activate();
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      single_task_termination = ApUserEvent::NO_AP_USER_EVENT;
      concurrent_fence_event = ApEvent::NO_AP_EVENT;
      copy_fill_priority = 0;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      selected_variant = 0;
      task_priority = 0;
      perform_postmap = false;
      first_mapping = true;
      execution_context = NULL;
      remote_trace_recorder = NULL;
      shard_manager = NULL;
      leaf_cached = false;
      inner_cached = false;
    }

    //--------------------------------------------------------------------------
    void SingleTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, DEACTIVATE_SINGLE_CALL);
      TaskOp::deactivate(freeop);
      target_processors.clear();
      physical_instances.clear();
      region_preconditions.clear();
      source_instances.clear();
      future_memories.clear();
      virtual_mapped.clear();
      no_access_regions.clear();
      intra_space_mapping_dependences.clear();
      map_applied_conditions.clear();
      task_completion_effects.clear();
      task_profiling_requests.clear();
      copy_profiling_requests.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
      untracked_valid_regions.clear();
      if ((execution_context != NULL) && 
          execution_context->remove_base_gc_ref(SINGLE_TASK_REF))
        delete execution_context; 
      if ((shard_manager != NULL) && 
          shard_manager->remove_base_gc_ref(SINGLE_TASK_REF))
        delete shard_manager;
#ifdef DEBUG_LEGION
      premapped_instances.clear();
      assert(remote_trace_recorder == NULL);
#endif
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_leaf(void) const
    //--------------------------------------------------------------------------
    {
      if (!leaf_cached)
      {
        VariantImpl *var = runtime->find_variant_impl(task_id,selected_variant);
        is_leaf_result = var->is_leaf();
        leaf_cached = true;
      }
      return is_leaf_result;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_inner(void) const
    //--------------------------------------------------------------------------
    {
      if (!inner_cached)
      {
        VariantImpl *var = runtime->find_variant_impl(task_id,selected_variant);
        is_inner_result = var->is_inner();
        inner_cached = true;
      }
      return is_inner_result;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_created_region(unsigned index) const
    //--------------------------------------------------------------------------
    {
      return (index >= get_region_count());
    }

    //--------------------------------------------------------------------------
    void SingleTask::update_no_access_regions(void)
    //--------------------------------------------------------------------------
    {
      no_access_regions.resize(logical_regions.size());
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
        no_access_regions[idx] = IS_NO_ACCESS(logical_regions[idx]) || 
                                  logical_regions[idx].privilege_fields.empty();
    } 

    //--------------------------------------------------------------------------
    void SingleTask::clone_single_from(SingleTask *rhs)
    //--------------------------------------------------------------------------
    {
      this->clone_task_op_from(rhs, this->target_proc, 
                               false/*stealable*/, true/*duplicate*/);
      this->index_point = rhs->index_point;
      this->virtual_mapped = rhs->virtual_mapped;
      this->no_access_regions = rhs->no_access_regions;
      this->target_processors = rhs->target_processors;
      this->physical_instances = rhs->physical_instances;
      this->intra_space_mapping_dependences = 
        rhs->intra_space_mapping_dependences;
      // no need to copy the control replication map
      this->selected_variant  = rhs->selected_variant;
      this->task_priority     = rhs->task_priority;
      this->shard_manager     = rhs->shard_manager;
      if (this->shard_manager != NULL)
        this->shard_manager->add_base_gc_ref(SINGLE_TASK_REF);
      // For now don't copy anything else below here
      // In the future we may need to copy the profiling requests
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_single_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_SINGLE_TASK_CALL);
      RezCheck z(rez);
      pack_base_task(rez, target);
      if (is_origin_mapped())
      {
        rez.serialize(selected_variant);
        rez.serialize(task_priority);
        rez.serialize<size_t>(target_processors.size());
        for (unsigned idx = 0; idx < target_processors.size(); idx++)
          rez.serialize(target_processors[idx]);
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
        {
          rez.serialize<bool>(virtual_mapped[idx]);
          if (virtual_mapped[idx])
            version_infos[idx].pack_equivalence_sets(rez);
        }
        rez.serialize(single_task_termination);
        rez.serialize<size_t>(physical_instances.size());
        for (unsigned idx = 0; idx < physical_instances.size(); idx++)
          physical_instances[idx].pack_references(rez);
        rez.serialize<size_t>(region_preconditions.size());
        for (unsigned idx = 0; idx < region_preconditions.size(); idx++)
          rez.serialize(region_preconditions[idx]);
        rez.serialize<size_t>(future_memories.size());
        for (unsigned idx = 0; idx < future_memories.size(); idx++)
          rez.serialize(future_memories[idx]);
        rez.serialize<size_t>(task_profiling_requests.size());
        for (unsigned idx = 0; idx < task_profiling_requests.size(); idx++)
          rez.serialize(task_profiling_requests[idx]);
        rez.serialize<size_t>(copy_profiling_requests.size());
        for (unsigned idx = 0; idx < copy_profiling_requests.size(); idx++)
          rez.serialize(copy_profiling_requests[idx]);
        if (!task_profiling_requests.empty() || !copy_profiling_requests.empty())
          rez.serialize(profiling_priority);
        rez.serialize<size_t>(untracked_valid_regions.size());
        for (unsigned idx = 0; idx < untracked_valid_regions.size(); idx++)
          rez.serialize(untracked_valid_regions[idx]); 
        rez.serialize(concurrent_fence_event);
      }
      else
      { 
        rez.serialize<size_t>(intra_space_mapping_dependences.size());
        for (unsigned idx = 0;
              idx < intra_space_mapping_dependences.size(); idx++)
          rez.serialize(intra_space_mapping_dependences[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::unpack_single_task(Deserializer &derez,
                                        std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, UNPACK_SINGLE_TASK_CALL);
      DerezCheck z(derez);
      unpack_base_task(derez, ready_events);
      if (map_origin)
      {
        derez.deserialize(selected_variant);
        derez.deserialize(task_priority);
        size_t num_target_processors;
        derez.deserialize(num_target_processors);
        target_processors.resize(num_target_processors);
        for (unsigned idx = 0; idx < num_target_processors; idx++)
          derez.deserialize(target_processors[idx]);
        virtual_mapped.resize(logical_regions.size());
        version_infos.resize(logical_regions.size());
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
        {
          bool result;
          derez.deserialize(result);
          virtual_mapped[idx] = result;
          if (result)
            version_infos[idx].unpack_equivalence_sets(derez, runtime, 
                                                       ready_events);
        }
        derez.deserialize(single_task_termination);
        size_t num_phy;
        derez.deserialize(num_phy);
        physical_instances.resize(num_phy);
        for (unsigned idx = 0; idx < num_phy; idx++)
          physical_instances[idx].unpack_references(runtime,
                                                    derez, ready_events);
        size_t num_pre;
        derez.deserialize(num_pre);
        region_preconditions.resize(num_pre);
        for (unsigned idx = 0; idx < num_pre; idx++)
          derez.deserialize(region_preconditions[idx]);
        size_t num_future_memories;
        derez.deserialize(num_future_memories);
        future_memories.resize(num_future_memories);
        for (unsigned idx = 0; idx < num_future_memories; idx++)
          derez.deserialize(future_memories[idx]);
        size_t num_task_requests;
        derez.deserialize(num_task_requests);
        if (num_task_requests > 0)
        {
          task_profiling_requests.resize(num_task_requests);
          for (unsigned idx = 0; idx < num_task_requests; idx++)
            derez.deserialize(task_profiling_requests[idx]);
        }
        size_t num_copy_requests;
        derez.deserialize(num_copy_requests);
        if (num_copy_requests > 0)
        {
          copy_profiling_requests.resize(num_copy_requests);
          for (unsigned idx = 0; idx < num_copy_requests; idx++)
            derez.deserialize(copy_profiling_requests[idx]);
        }
        if (!task_profiling_requests.empty() || 
            !copy_profiling_requests.empty())
          derez.deserialize(profiling_priority);
        size_t num_untracked_valid_regions;
        derez.deserialize(num_untracked_valid_regions);
        untracked_valid_regions.resize(num_untracked_valid_regions);
        for (unsigned idx = 0; idx < num_untracked_valid_regions; idx++)
          derez.deserialize(untracked_valid_regions[idx]); 
        derez.deserialize(concurrent_fence_event);
      }
      else
      {
        size_t num_intra_space_dependences;
        derez.deserialize(num_intra_space_dependences);
        intra_space_mapping_dependences.resize(num_intra_space_dependences);
        for (unsigned idx = 0; idx < num_intra_space_dependences; idx++)
          derez.deserialize(intra_space_mapping_dependences[idx]);
      }
      update_no_access_regions();
    } 

    //--------------------------------------------------------------------------
    void SingleTask::shard_off(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      // Still need this to record that this operation is done for LegionSpy
      LegionSpy::log_operation_events(unique_op_id, 
          ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
#endif
      // Do the stuff to record that this is mapped and executed
      complete_mapping(mapped_precondition);
      complete_execution();
      trigger_children_complete();
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void SingleTask::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, TRIGGER_SINGLE_CALL);
      if (is_remote())
      {
        if (distribute_task())
        {
          // Still local
          if (is_origin_mapped())
          {
            // Remote and origin mapped means
            // we were already mapped so we can
            // just launch the task
            launch_task();
          }
          else
          {
            // Remote but still need to map
            if (is_replicable())
            {
              if (replicate_task())
                return;
              replicate = false;
            }
            const RtEvent done_mapping = perform_mapping();
            if (done_mapping.exists() && !done_mapping.has_triggered())
              defer_launch_task(done_mapping);
            else
              launch_task();
          }
        }
        // otherwise it was sent away
      }
      else
      {
        // See if we have a must epoch in which case
        // we can simply record ourselves and we are done
        if (must_epoch == NULL)
        {
#ifdef DEBUG_LEGION
          assert(target_proc.exists());
#endif
          // See if this task is going to be sent
          // remotely in which case we need to do the
          // mapping now, otherwise we can defer it
          // until the task ends up on the target processor
          if (is_origin_mapped())
          {
            if (first_mapping)
            {
              first_mapping = false;
              const RtEvent done_mapping = perform_mapping();
              if (!done_mapping.exists() || done_mapping.has_triggered())
              {
                if (distribute_task())
                  launch_task();
              }
              else
                defer_distribute_task(done_mapping);
            }
            else if (distribute_task())
              launch_task();
          }
          else
          {
            if (distribute_task())
            {
              // Still local so try mapping and launching
              if (is_replicable())
              {
                if (replicate_task())
                  return;
                replicate = false;
              }
              const RtEvent done_mapping = perform_mapping();
              if (!done_mapping.exists() || done_mapping.has_triggered())
                launch_task();
              else
                defer_launch_task(done_mapping);
            }
          }
        }
        else
          must_epoch->register_single_task(this, must_epoch_index);
      }
    } 

    //--------------------------------------------------------------------------
    void SingleTask::perform_inlining(VariantImpl *variant,
                                const std::deque<InstanceSet> &parent_instances)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(parent_instances.size() == regions.size());
#endif
      selected_variant = variant->vid;
      target_processors.push_back(current_proc);
      physical_instances = parent_instances;
      virtual_mapped.resize(regions.size());
      no_access_regions.resize(regions.size());
      region_preconditions.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        virtual_mapped[idx] = false;
        no_access_regions[idx] = IS_NO_ACCESS(regions[idx]);
        region_preconditions[idx] = ApEvent::NO_AP_EVENT;
      }
      complete_mapping();
      // Now we can launch this task right inline in this thread
      launch_task(true/*inline*/); 
    }

    //--------------------------------------------------------------------------
    RtEvent SingleTask::perform_versioning_analysis(const bool post_mapper)
    //--------------------------------------------------------------------------
    {
      if (is_replaying())
        return RtEvent::NO_RT_EVENT;
      // If we're remote and origin mapped, then we are already done
      if (is_remote() && is_origin_mapped())
        return RtEvent::NO_RT_EVENT;
#ifdef DEBUG_LEGION
      assert(version_infos.empty() || 
              (version_infos.size() == get_region_count()));
#endif
      version_infos.resize(get_region_count());
      std::set<RtEvent> ready_events;
      std::vector<RtEvent> output_events;
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
      {
        if (no_access_regions[idx] || (post_mapper && virtual_mapped[idx]))
          continue;
        VersionInfo &version_info = version_infos[idx];
        if (version_info.has_version_info())
          continue;
        const RegionRequirement &req = logical_regions[idx];
        if ((regions.size() <= idx) && !is_output_valid(idx-regions.size()))
        {
          RtEvent output_ready;
          runtime->forest->perform_versioning_analysis(this, idx, req,
              version_info, ready_events, &output_ready);
#ifdef DEBUG_LEGION
          assert(output_ready.exists());
#endif
          output_events.push_back(output_ready);
        }
        else
          runtime->forest->perform_versioning_analysis(this, idx,
              req, version_info, ready_events, NULL/*output region*/,
              IS_COLLECTIVE(req) || std::binary_search(
                check_collective_regions.begin(),
                check_collective_regions.end(), idx));
      }
      if (!output_events.empty())
        record_output_registered(
            Runtime::merge_events(output_events), ready_events);
      if (!ready_events.empty())
        return Runtime::merge_events(ready_events);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void SingleTask::initialize_map_task_input(Mapper::MapTaskInput &input,
                                               Mapper::MapTaskOutput &output,
                                               MustEpochOp *must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INITIALIZE_MAP_TASK_CALL);
      // Do the traversals for all the regions and find
      // their valid instances, then fill in the mapper input structure
      input.valid_instances.resize(regions.size());
      input.valid_collectives.resize(regions.size());
      input.shard_processor = Processor::NO_PROC;
      input.shard_variant = 0;
      output.chosen_instances.resize(regions.size());
      output.source_instances.resize(regions.size());
      output.output_targets.resize(output_regions.size());
      output.output_constraints.resize(output_regions.size());
      // If we have must epoch owner, we have to check for any 
      // constrained mappings which must be heeded
      if (must_epoch_owner != NULL)
        must_epoch_owner->must_epoch_map_task_callback(this, input, output);
      std::set<Memory> visible_memories;
      runtime->machine.get_visible_memories(target_proc, visible_memories);
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Skip any NO_ACCESS or empty privilege field regions
        if (IS_NO_ACCESS(regions[idx]) || regions[idx].privilege_fields.empty())
          continue;
        // See if we've already got an output from a must-epoch mapping
        if (!output.chosen_instances[idx].empty())
        {
#ifdef DEBUG_LEGION
          assert(must_epoch_owner != NULL);
#endif
          // We can skip this since we already know the result
          continue;
        }
        if (request_valid_instances && 
            (regions[idx].privilege != LEGION_REDUCE))
        {
          InstanceSet current_valid;
          FieldMaskSet<ReplicatedView> collectives;
          runtime->forest->physical_premap_region(this, idx, regions[idx],
                version_infos[idx], current_valid, 
                collectives, map_applied_conditions);
          if (regions[idx].is_no_access())
            prepare_for_mapping(current_valid, collectives,
                input.valid_instances[idx], input.valid_collectives[idx]);
          else
            prepare_for_mapping(current_valid, collectives, visible_memories,
                input.valid_instances[idx], input.valid_collectives[idx]);
        }
      }
#ifdef DEBUG_LEGION
      // Save the inputs for premapped regions so we can check them later
      if (!input.premapped_regions.empty())
      {
        for (std::vector<unsigned>::const_iterator it = 
              input.premapped_regions.begin(); it !=
              input.premapped_regions.end(); it++)
          premapped_instances[*it] = output.chosen_instances[*it];
      }
#endif
      // Prepare the output too
      output.chosen_variant = 0;
      output.postmap_task = false;
      output.task_priority = 0;
      output.postmap_task = false;
    }

    //--------------------------------------------------------------------------
    void SingleTask::finalize_map_task_output(Mapper::MapTaskInput &input,
                                              Mapper::MapTaskOutput &output,
                                              MustEpochOp *must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, FINALIZE_MAP_TASK_CALL);
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      // first check the processors to make sure they are all on the
      // same node and of the same kind, if we know we have a must epoch
      // owner then we also know there is only one valid choice
      if (must_epoch_owner == NULL)
      {
        if (output.target_procs.empty())
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_EMPTY_OUTPUT_TARGET,
                          "Empty output target_procs from call to 'map_task' "
                          "by mapper %s for task %s (ID %lld). Adding the "
                          "'target_proc' " IDFMT " as the default.",
                          mapper->get_mapper_name(), get_task_name(),
                          get_unique_id(), this->target_proc.id);
          output.target_procs.push_back(this->target_proc);
        }
        else if (output.target_procs.size() > 1)
        {
          if (concurrent_task)
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                "Mapper %s provided multiple target processors as output "
                "from 'map_task' for task %s (UID %lld) which was launched "
                "in a concurrent index space task launch. Mappers are only "
                "permitted to specify a single target processor for mapping "
                "tasks in concurrent index space task launches.",
                mapper->get_mapper_name(), get_task_name(), get_unique_id())
          else if (runtime->separate_runtime_instances)
            // Ignore additional processors in separate runtime instances
            output.target_procs.resize(1);
        } 
        if (!runtime->unsafe_mapper)
          validate_target_processors(output.target_procs);
        // Save the target processors from the output
        target_processors = output.target_procs;
        target_proc = target_processors.front();
      }
      else
      {
        if (output.target_procs.size() > 1)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_SPURIOUS_TARGET,
                          "Ignoring spurious additional target processors "
                          "requested in 'map_task' for task %s (ID %lld) "
                          "by mapper %s because task is part of a must "
                          "epoch launch.", get_task_name(), get_unique_id(),
                          mapper->get_mapper_name());
        }
        if (!output.target_procs.empty() && 
                 (output.target_procs[0] != this->target_proc))
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_PROCESSOR_REQUEST,
                          "Ignoring processor request of " IDFMT " for "
                          "task %s (ID %lld) by mapper %s because task "
                          "has already been mapped to processor " IDFMT
                          " as part of a must epoch launch.", 
                          output.target_procs[0].id, get_task_name(), 
                          get_unique_id(), mapper->get_mapper_name(),
                          this->target_proc.id);
        }
        // Only one valid choice in this case, ignore everything else
        target_processors.push_back(this->target_proc);
      }
      // If we had any future mapping outputs, we can grab them
      if (!futures.empty())
      {
        future_memories.swap(output.future_locations);
        if (futures.size() < future_memories.size())
          future_memories.resize(futures.size());
        // Check to make sure that they are all on the same address
        // space as the target processor(s)
        const AddressSpaceID target_space = this->target_proc.address_space();
        for (unsigned idx = 0; idx < future_memories.size(); idx++)
        {
          if (!future_memories[idx].exists())
            continue;
          if (future_memories[idx].address_space() != target_space)
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                "Invalid mapper output from invocation of '%s' on mapper %s "
                "when mapping task %s (UID %lld). Mapper attempted to map "
                "future %d to memory " IDFMT " in address space "
                "%d which is not the same as address space %d of the target "
                "processor " IDFMT ". Mapped futures must be in the same "
                "address space as the target processor for task mappings.",
                "map_task", mapper->get_mapper_name(), get_task_name(),
                get_unique_id(), idx, future_memories[idx].id, 
                future_memories[idx].address_space(), target_space,
                this->target_proc.id)
          // Request the future memories be created
          const RtEvent future_mapped =
            futures[idx].impl->request_application_instance(
              future_memories[idx], this, unique_op_id, target_space);
          if (future_mapped.exists())
            map_applied_conditions.insert(future_mapped); 
        }
        // Handle any unmapped futures too
        Memory target_memory = Memory::NO_MEMORY;
        for (unsigned idx = future_memories.size(); idx < futures.size(); idx++)
        {
          if (!target_memory.exists())
          {
            if (target_space != runtime->address_space)
              target_memory = runtime->find_local_memory(this->target_proc,
                                                        Memory::SYSTEM_MEM);
            else
              target_memory = runtime->runtime_system_memory;
          }
          future_memories.push_back(target_memory);
          const RtEvent future_mapped =
            futures[idx].impl->request_application_instance(
              target_memory, this, unique_op_id, target_space);
          if (future_mapped.exists())
            map_applied_conditions.insert(future_mapped);
        }
      }
      // Sort out any profiling requests that we need to perform
      if (!output.task_prof_requests.empty())
      {
        profiling_priority = output.profiling_priority;
        // If we do any legion specific checks, make sure we ask
        // Realm for the proc profiling info so that we can get
        // a callback to report our profiling information
        bool has_proc_request = false;
        // Filter profiling requests into those for copies and the actual task
        for (std::set<ProfilingMeasurementID>::const_iterator it = 
              output.task_prof_requests.requested_measurements.begin(); it !=
              output.task_prof_requests.requested_measurements.end(); it++)
        {
          if ((*it) > Mapping::PMID_LEGION_FIRST)
          {
            // If we haven't seen a proc usage yet, then add it
            // to the realm requests to ensure we get a callback
            // for this task. We know we'll see it before this
            // because the measurement IDs are in order
            if (!has_proc_request)
              task_profiling_requests.push_back(
                  (ProfilingMeasurementID)Realm::PMID_OP_PROC_USAGE);
            // These are legion profiling requests and currently
            // are only profiling task information
            task_profiling_requests.push_back(*it);
            continue;
          }
          switch ((Realm::ProfilingMeasurementID)*it)
          {
            case Realm::PMID_OP_PROC_USAGE:
              has_proc_request = true; // Then fall through
            case Realm::PMID_OP_STATUS:
            case Realm::PMID_OP_BACKTRACE:
            case Realm::PMID_OP_TIMELINE:
            case Realm::PMID_OP_TIMELINE_GPU:
            case Realm::PMID_PCTRS_CACHE_L1I:
            case Realm::PMID_PCTRS_CACHE_L1D:
            case Realm::PMID_PCTRS_CACHE_L2:
            case Realm::PMID_PCTRS_CACHE_L3:
            case Realm::PMID_PCTRS_IPC:
            case Realm::PMID_PCTRS_TLB:
            case Realm::PMID_PCTRS_BP:
              {
                // Just task
                task_profiling_requests.push_back(*it);
                break;
              }
            default:
              {
                REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_PROFILING,
                              "Mapper %s requested a profiling "
                    "measurement of type %d which is not applicable to "
                    "task %s (UID %lld) and will be ignored.",
                    mapper->get_mapper_name(), *it, get_task_name(),
                    get_unique_id());
              }
          }
        }
#ifdef DEBUG_LEGION
        assert(!profiling_reported.exists());
        assert(outstanding_profiling_requests == 0);
#endif
        profiling_reported = Runtime::create_rt_user_event();
        // Increment the number of profiling responses here since we
        // know that we're going to get one for launching the task
        // No need for the lock since no outstanding physical analyses
        // can be running yet
        outstanding_profiling_requests = 1;
      }
      if (!output.copy_prof_requests.empty())
      {
        filter_copy_request_kinds(mapper, 
            output.copy_prof_requests.requested_measurements,
            copy_profiling_requests, true/*warn*/);
        profiling_priority = output.profiling_priority;
        if (!profiling_reported.exists())
          profiling_reported = Runtime::create_rt_user_event();
      }
      // See whether the mapper picked a variant or a generator
      VariantImpl *variant_impl = NULL;
      if (output.chosen_variant > 0)
        variant_impl = runtime->find_variant_impl(task_id, 
                                output.chosen_variant, true/*can fail*/);
      else // TODO: invoke a generator if one exists
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of '%s' on "
                      "mapper %s. Mapper specified an invalid task variant "
                      "of ID 0 for task %s (ID %lld), but Legion does not yet "
                      "support task generators.", "map_task", 
                      mapper->get_mapper_name(), 
                      get_task_name(), get_unique_id())
      if (variant_impl == NULL)
        // If we couldn't find or make a variant that is bad
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of '%s' on "
                      "mapper %s. Mapper failed to specify a valid "
                      "task variant or generator capable of create a variant "
                      "implementation of task %s (ID %lld).",
                      "map_task", mapper->get_mapper_name(), get_task_name(),
                      get_unique_id())
      // Record the future output size
      handle_future_size(variant_impl->return_type_size,
          variant_impl->has_return_type_size, map_applied_conditions);
      if (is_recording() && !variant_impl->has_return_type_size)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
            "Invalid mapper output from invocation of '%s' on mapper %s. "
            "Mapper selected task variant %d when mapping task %s (UID %lld) "
            "being recorded for trace %d in parent task %s (UID %lld). "
            "However this variant does not specify a static upper bound "
            "future size. All tasks recorded as part of a trace must use "
            "variants with statically known future result sizes.", "map_task",
            mapper->get_mapper_name(), output.chosen_variant, get_task_name(),
            get_unique_id(), trace->tid, parent_ctx->get_task_name(),
            parent_ctx->get_unique_id())
      // Save variant validation until we know which instances we'll be using 
#ifdef DEBUG_LEGION
      // Check to see if any premapped region mappings changed
      if (!premapped_instances.empty())
      {
        for (std::map<unsigned,std::vector<Mapping::PhysicalInstance> >::
              const_iterator it = premapped_instances.begin(); it !=
              premapped_instances.end(); it++)
        {
          if (it->second.size() != output.chosen_instances[it->first].size())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper modified the premapped output "
                        "for region requirement %d of task %s (ID %lld).",
                        "map_task", mapper->get_mapper_name(), it->first,
                        get_task_name(), get_unique_id())
          for (unsigned idx = 0; idx < it->second.size(); idx++)
            if (it->second[idx] != output.chosen_instances[it->first][idx])
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper modified the premapped output "
                        "for region requirement %d of task %s (ID %lld).",
                        "map_task", mapper->get_mapper_name(), it->first,
                        get_task_name(), get_unique_id())
        }
      }
#endif
      // fill in virtual_mapped
      virtual_mapped.resize(logical_regions.size(),false);
      // Convert all the outputs into our set of physical instances and
      // validate them by checking the following properites:
      // - all are either pure virtual or pure physical 
      // - no missing fields
      // - all satisfy the region requirement
      // - all are visible from all the target processors
      physical_instances.resize(logical_regions.size());
      source_instances.resize(logical_regions.size());
      // If we're doing safety checks, we need the set of memories
      // visible from all the target processors
      std::set<Memory> visible_memories;
      if (!runtime->unsafe_mapper)
      {
        if (target_processors.size() > 1)
        {
          // If we have multiple processor, we want the set of 
          // memories visible to all of them
          Machine::MemoryQuery visible_query(runtime->machine);
          for (std::vector<Processor>::const_iterator it = 
                target_processors.begin(); it != target_processors.end(); it++)
            visible_query.has_affinity_to(*it);
          for (Machine::MemoryQuery::iterator it = visible_query.begin();
                it != visible_query.end(); it++)
            visible_memories.insert(*it);
        }
        else
          runtime->find_visible_memories(target_proc, visible_memories);
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Skip any NO_ACCESS or empty privilege field regions
        if (no_access_regions[idx])
          continue; 
        // Do the conversion
        InstanceSet &result = physical_instances[idx];
        RegionTreeID bad_tree = 0;
        std::vector<FieldID> missing_fields;
        std::vector<PhysicalManager*> unacquired;
        bool free_acquired = false;
        std::map<PhysicalManager*,unsigned> *acquired = NULL;
        // Get the acquired instances only if we are checking
        if (!runtime->unsafe_mapper)
        {
          if (this->must_epoch != NULL)
          {
            acquired = new std::map<PhysicalManager*,unsigned>(
                                  *get_acquired_instances_ref());
            free_acquired = true;
            // Merge the must epoch owners acquired instances too 
            // if we need to check for all our instances being acquired
            std::map<PhysicalManager*,unsigned> *epoch_acquired = 
              this->must_epoch->get_acquired_instances_ref();
            if (epoch_acquired != NULL)
              acquired->insert(epoch_acquired->begin(), epoch_acquired->end());
          }
          else
            acquired = get_acquired_instances_ref();
        }
        // Convert any sources first
        if (!output.source_instances[idx].empty())
          runtime->forest->physical_convert_sources(this, regions[idx],
              output.source_instances[idx], source_instances[idx], acquired);
        int composite_idx = 
          runtime->forest->physical_convert_mapping(this, regions[idx],
                output.chosen_instances[idx], result, bad_tree, missing_fields,
                acquired, unacquired, !runtime->unsafe_mapper);
        if (free_acquired)
          delete acquired;
        if (bad_tree > 0)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper specified an instance from region "
                        "tree %d for use with region requirement %d of task "
                        "%s (ID %lld) whose region is from region tree %d.",
                        "map_task",mapper->get_mapper_name(), bad_tree,
                        idx, get_task_name(), get_unique_id(),
                        regions[idx].region.get_tree_id())
        if (!missing_fields.empty())
        {
          for (std::vector<FieldID>::const_iterator it = 
                missing_fields.begin(); it != missing_fields.end(); it++)
          {
            const void *name; size_t name_size;
            if (!runtime->retrieve_semantic_information(
                regions[idx].region.get_field_space(), *it, 
                LEGION_NAME_SEMANTIC_TAG, name, name_size, 
                true/*can fail*/, false))
	          name = "(no name)";
              log_run.error("Missing instance for field %s (FieldID: %d)",
                          static_cast<const char*>(name), *it);
          }
          REPORT_LEGION_ERROR(ERROR_MISSING_INSTANCE_FIELD,
                        "Invalid mapper output from invocation of '%s' on "
                        "mapper %s. Mapper failed to specify an instance for "
                        "%zd fields of region requirement %d on task %s "
                        "(ID %lld). The missing fields are listed below.",
                        "map_task", mapper->get_mapper_name(), 
                        missing_fields.size(), idx, get_task_name(), 
                        get_unique_id())
          
        }
        if (!unacquired.empty())
        {
          std::map<PhysicalManager*,unsigned> *acquired_instances = 
            get_acquired_instances_ref();
          for (std::vector<PhysicalManager*>::const_iterator it = 
                unacquired.begin(); it != unacquired.end(); it++)
          {
            if (acquired_instances->find(*it) == acquired_instances->end())
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from 'map_task' "
                            "invocation on mapper %s. Mapper selected "
                            "physical instance for region requirement "
                            "%d of task %s (ID %lld) which has already "
                            "been collected. If the mapper had properly "
                            "acquired this instance as part of the mapper "
                            "call it would have detected this. Please "
                            "update the mapper to abide by proper mapping "
                            "conventions.", mapper->get_mapper_name(),
                            idx, get_task_name(), get_unique_id())
          }
          // Event if we did successfully acquire them, still issue the warning
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_FAILED_ACQUIRE,
                          "mapper %s failed to acquire instances "
                          "for region requirement %d of task %s (ID %lld) "
                          "in 'map_task' call. You may experience "
                          "undefined behavior as a consequence.",
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id())
        }
        // See if they want a virtual mapping
        if (composite_idx >= 0)
        {
          // Everything better be all virtual or all real
          if (result.size() > 1)
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of '%s' on "
                          "mapper %s. Mapper specified mixed composite and "
                          "concrete instances for region requirement %d of "
                          "task %s (ID %lld). Only full concrete instances "
                          "or a single composite instance is supported.",
                          "map_task", mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id())
          if (IS_REDUCE(regions[idx]))
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of '%s' on "
                          "mapper %s. Illegal composite mapping requested on "
                          "region requirement %d of task %s (UID %lld) which "
                          "has only reduction privileges.", 
                          "map_task", mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id())
          if (!IS_EXCLUSIVE(regions[idx]))
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of '%s' on "
                          "mapper %s. Illegal composite instance requested "
                          "on region requirement %d of task %s (ID %lld) "
                          "which has a relaxed coherence mode. Virtual "
                          "mappings are only permitted for exclusive "
                          "coherence.", "map_task", mapper->get_mapper_name(),
                          idx, get_task_name(), get_unique_id())
          virtual_mapped[idx] = true;
        }
        log_mapping_decision(idx, regions[idx], physical_instances[idx]);
        // Skip checks if the mapper promises it is safe
        if (runtime->unsafe_mapper)
          continue;
        // If this is anything other than a virtual mapping, check that
        // the instances align with the privileges
        if (!virtual_mapped[idx])
        {
          std::vector<LogicalRegion> regions_to_check(1, regions[idx].region);
          for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
          {
            PhysicalManager *manager = result[idx2].get_physical_manager();
            if (!manager->meets_regions(regions_to_check))
              // Doesn't satisfy the region requirement
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from invocation of '%s' on "
                            "mapper %s. Mapper specified instance that does "
                            "not meet region requirement %d for task %s "
                            "(ID %lld). The index space for the instance has "
                            "insufficient space for the requested logical "
                            "region.", "map_task", mapper->get_mapper_name(),
                            idx, get_task_name(), get_unique_id())
          }
          if (!regions[idx].is_no_access() &&
              !variant_impl->is_no_access_region(idx))
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
            {
              const Memory mem = result[idx2].get_memory();
              if (visible_memories.find(mem) == visible_memories.end())
                // Not visible from all target processors
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                              "Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper selected an instance for "
                              "region requirement %d in memory " IDFMT " "
                              "which is not visible from the target processors "
                              "for task %s (ID %lld).", "map_task", 
                              mapper->get_mapper_name(), idx, mem.id, 
                              get_task_name(), get_unique_id())
            }
          }
          // If this is a reduction region requirement make sure all the 
          // managers are reduction instances with the right reduction ops
          if (IS_REDUCE(regions[idx]))
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
            {
              PhysicalManager *manager = result[idx2].get_physical_manager();
              if (!manager->is_reduction_manager())
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                              "Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper failed to choose a "
                              "specialized reduction instance for region "
                              "requirement %d of task %s (ID %lld) which has "
                              "reduction privileges.", "map_task", 
                              mapper->get_mapper_name(), idx,
                              get_task_name(), get_unique_id())
              else if (manager->redop != regions[idx].redop)
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                              "Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper failed selected a "
                              "specialized reduction instance with reduction "
                              "operator %d for region requirement %d of task "
                              "%s (ID %lld) which has reduction privileges "
                              "on a different reduction operator %d.", 
                              "map_task", mapper->get_mapper_name(), 
                              manager->redop, idx, get_task_name(), 
                              get_unique_id(), regions[idx].redop)
            }
          }
          else
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
              if (result[idx2].get_manager()->is_reduction_manager())
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                              "Invalid mapper output from invocation of '%s' "
                              "on mapper %s. Mapper selected illegal "
                              "specialized reduction instance for region "
                              "requirement %d of task %s (ID %lld) which "
                              "does not have reduction privileges.", "map_task",
                              mapper->get_mapper_name(), idx, 
                              get_task_name(), get_unique_id())
          }
        }
      }

      if (!output_regions.empty())
      {
        // Now we prepare output instances
        if (!runtime->unsafe_mapper)
          for (unsigned idx = 0; idx < output_regions.size(); idx++)
          {
            Memory target = output.output_targets[idx];
            if (!target.exists() ||
                visible_memories.find(target) == visible_memories.end())
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from invocation of '%s' "
                            "on mapper %s. Mapper selected invalid "
                            "target memory " IDFMT " for output region "
                            "requirement %d of task %s (ID %lld).", "map_task",
                            mapper->get_mapper_name(), target.id, idx,
                            get_task_name(), get_unique_id())
          }
        const size_t output_offset = regions.size();
        for (unsigned idx = 0; idx < output_regions.size(); idx++)
        {
          prepare_output_instance(idx,
                                  physical_instances[output_offset + idx],
                                  output_regions[idx],
                                  output.output_targets[idx],
                                  output.output_constraints[idx]);
          log_mapping_decision(output_offset+idx, output_regions[idx],
                               physical_instances[output_offset + idx]);
        }
      }
      // If the variant has padded fields we need to get the atomic locks
      if (variant_impl->needs_padding)
        variant_impl->find_padded_locks(this, regions, physical_instances);
      // Now that we have our physical instances we can validate the variant
      if (!runtime->unsafe_mapper)
      {
#ifdef DEBUG_LEGION
        assert(!target_processors.empty());
#endif
        validate_variant_selection(mapper, variant_impl,
            target_processors.front().kind(), physical_instances, "map_task");
      }
      // Record anything else that needs to be recorded 
      selected_variant = output.chosen_variant;
      task_priority = output.task_priority;
      perform_postmap = output.postmap_task;
      if (!output.untracked_valid_regions.empty())
      {
        for (std::set<unsigned>::const_iterator it =
              output.untracked_valid_regions.begin(); it != 
              output.untracked_valid_regions.end(); it++)
        {
          // Remove it if it is too big or is not read-only
          if ((*it >= regions.size()) || !IS_READ_ONLY(regions[*it]))
          {
            if (*it < regions.size())
              REPORT_LEGION_WARNING(LEGION_WARNING_NON_READ_ONLY_UNTRACK_VALID,
                  "Ignoring request by mapper %s to not track valid instances "
                  "for region requirement %d of task %s (UID %lld) because "
                  "region requirement does not have read-only privileges.",
                  mapper->get_mapper_name(), *it, 
                  get_task_name(), unique_op_id)
          }
          else
            untracked_valid_regions.push_back(*it);
        }
      } 
    }

    //--------------------------------------------------------------------------
    void SingleTask::prepare_output_instance(unsigned index,
                                             InstanceSet &instance_set,
                                             const RegionRequirement &req,
                                             Memory target,
                                             const LayoutConstraintSet &c)
    //--------------------------------------------------------------------------
    {
      MemoryManager *memory_manager = runtime->find_memory_manager(target);

      std::map<PhysicalManager*,unsigned> *acquired_instances =
        get_acquired_instances_ref();

      LayoutConstraintSet constraints;
      constraints.add_constraint(MemoryConstraint(target.kind()))
        .add_constraint(
            SpecializedConstraint(LEGION_AFFINE_SPECIALIZE, 0, false, true))
        .add_constraint(c.ordering_constraint);

#ifdef DEBUG_LEGION
      const std::vector<DimensionKind> &ordering =
        constraints.ordering_constraint.ordering;
      if (ordering.empty())
      {
        REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_CONSTRAINTS,
          "An ordering constraint must be specified for each output "
          "region, but the mapper did not specify any ordering constraint "
          "for output region %u of task %s (UID: %lld).",
          index, get_task_name(), get_unique_op_id());
      }
      else if (static_cast<int>(ordering.size()) != req.region.get_dim() + 1)
      {
        REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_CONSTRAINTS,
          "The mapper chose an ordering constraint with %d dimensions "
          "for output region %u of task %s (UID: %lld), but the region has "
          "%d dimensions. Make sure you specify a correct ordering.",
          static_cast<int>(ordering.size()) - 1, index, get_task_name(),
          get_unique_op_id(), req.region.get_dim());
      }
      else
      {
        // TODO: For now we only allow SOA layout with either the C order
        // or the Fotran order for output instances.
        if (ordering.back() != LEGION_DIM_F)
        {
          REPORT_LEGION_FATAL(LEGION_FATAL_UNIMPLEMENTED_FEATURE,
            "Legion currently supports only the SOA layout for output regions, "
            "but output region %u of task %s (UID: %lld) is mapped to a "
            "non-SOA layout. Please update the mapper to use SOA layout "
            "for all output regions.",
            index, get_task_name(), get_unique_op_id());
        }
      }
#endif
      std::map<FieldID, std::pair<EqualityKind, size_t> > alignments;
      std::map<FieldID, off_t> offsets;

      for (std::vector<AlignmentConstraint>::const_iterator it =
           c.alignment_constraints.begin(); it !=
           c.alignment_constraints.end(); ++it)
      {
#ifdef DEBUG_LEGION
        assert(alignments.find(it->fid) == alignments.end());
#endif
        alignments[it->fid] = std::make_pair(it->eqk, it->alignment);
      }

      for (std::vector<OffsetConstraint>::const_iterator it =
           c.offset_constraints.begin(); it !=
           c.offset_constraints.end(); ++it)
      {
#ifdef DEBUG_LEGION
        assert(offsets.find(it->fid) == offsets.end());
#endif
        offsets[it->fid] = it->offset;
      }

      for (std::set<FieldID>::iterator it = req.privilege_fields.begin();
           it != req.privilege_fields.end(); ++it)
      {
        // Create a layout description with a single field
        std::vector<FieldID> fields(1, *it);
        constraints.field_constraint = FieldConstraint(fields, false, false);

        {
          std::map<FieldID, std::pair<EqualityKind, size_t> >::iterator finder =
            alignments.find(*it);
          if (finder != alignments.end())
            constraints.add_constraint(
              AlignmentConstraint(
                finder->first, finder->second.first, finder->second.second));
        }

        {
          std::map<FieldID, off_t>::iterator finder = offsets.find(*it);
          if (finder != offsets.end())
            constraints.add_constraint(
              OffsetConstraint(finder->first, finder->second));
        }

#ifdef DEBUG_LEGION
        assert(single_task_termination.exists());
#endif

        // Create a physical manager that is not bound to any instance
        PhysicalManager *manager =
          memory_manager->create_unbound_instance(req.region,
                                                  constraints,
                                                  single_task_termination,
                                                  map_id,
                                                  target_proc,
                                                  0/*priority*/);

        // Add an instance ref of the new manager to the instance set
        instance_set.add_instance(
            InstanceRef(manager, manager->layout->allocated_fields));

        // Add the manager to the map of acquired instances so that
        // later we can release it properly
        acquired_instances->insert(std::make_pair(manager, 1));

        constraints.alignment_constraints.clear();
        constraints.offset_constraints.clear();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_completion_effects(std::set<ApEvent> &effects, 
                                             bool tracing)
    //--------------------------------------------------------------------------
    {
      Operation::find_completion_effects(effects, tracing);
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      // If we're completed then we know we have all the completion effects
      // that we're ever going to have so we can just report them back
      if (!task_completion_effects.empty())
        effects.insert(task_completion_effects.begin(),
                       task_completion_effects.end());
    }

    //--------------------------------------------------------------------------
    void SingleTask::find_completion_effects(std::vector<ApEvent> &effects,
                                             bool tracing)
    //--------------------------------------------------------------------------
    {
      Operation::find_completion_effects(effects, tracing);
      AutoLock o_lock(op_lock,1,false/*exclusive*/);
      // If we're completed then we know we have all the completion effects
      // that we're ever going to have so we can just report them back
      if (!task_completion_effects.empty())
        effects.insert(effects.end(), task_completion_effects.begin(),
                       task_completion_effects.end());
    }

    //--------------------------------------------------------------------------
    void SingleTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      tpl->register_operation(this);
      tpl->get_mapper_output(this, selected_variant, task_priority,
          perform_postmap, target_processors, future_memories,
          physical_instances);
      // Then request any future mappings in advance
      if (!futures.empty())
      {
        for (unsigned idx = 0; idx < futures.size(); idx++)
        {
          const Memory memory = future_memories[idx];
          const RtEvent future_mapped =
            futures[idx].impl->request_application_instance(memory, this,
               unique_op_id, memory.address_space());
          if (future_mapped.exists())
            map_applied_conditions.insert(future_mapped);
        }
      }
      // Make sure to propagate any future sizes that we know about here
      if (!elide_future_return)
      {
        VariantImpl *variant_impl = 
          runtime->find_variant_impl(task_id, selected_variant);
#ifdef DEBUG_LEGION
        assert(variant_impl->has_return_type_size);
#endif
        // Record the future output size
        handle_future_size(variant_impl->return_type_size,
            variant_impl->has_return_type_size, map_applied_conditions);
      }
      if (!single_task_termination.exists())
        single_task_termination = Runtime::create_ap_user_event(NULL);
      set_origin_mapped(true); // it's like this was origin mapped
#ifdef DEBUG_LEGION
      // should only be replaying leaf tasks currently
      // until we figure out how to handle non-leaf tasks
      assert(is_leaf());
#endif
      if (is_leaf())
        handle_post_mapped(RtEvent::NO_RT_EVENT);
    }

    //--------------------------------------------------------------------------
    void SingleTask::handle_post_mapped(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
      if (!map_applied_conditions.empty())
      {
        if (mapped_precondition.exists())
          map_applied_conditions.insert(mapped_precondition);
        mapped_precondition = Runtime::merge_events(map_applied_conditions);
      }
      if (!acquired_instances.empty())
        mapped_precondition = release_nonempty_acquired_instances(
            mapped_precondition, acquired_instances);
      complete_mapping(mapped_precondition);
    }

    //--------------------------------------------------------------------------
    ApEvent SingleTask::replay_mapping(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(output_regions.empty());
      assert(single_task_termination.exists());
#endif
      virtual_mapped.resize(regions.size(), false);
      bool needs_reservations = false;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        InstanceSet &instances = physical_instances[idx];
        if (IS_NO_ACCESS(regions[idx]))
          continue;
        if (IS_ATOMIC(regions[idx]) || IS_REDUCE(regions[idx]))
          needs_reservations = true;
        if (instances.is_virtual_mapping())
          virtual_mapped[idx] = true;
        log_mapping_decision(idx, regions[idx], instances);
      }
      if (needs_reservations)
        // We group all reservations together anyway
        tpl->get_task_reservations(this, atomic_locks);
      return single_task_termination;
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_replicate_collective_versioning(unsigned index,
        unsigned parent_req_index, LegionMap<LogicalRegion,
            CollectiveVersioningBase::RegionVersioning> &to_perform)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_manager != NULL);
      assert(!IS_COLLECTIVE(regions[index]));
      assert(!std::binary_search(check_collective_regions.begin(),
            check_collective_regions.end(), index));
#endif
      // Bounce it back onto the shard manager to finalize
      shard_manager->finalize_replicate_collective_versioning(index,
          parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    void SingleTask::convert_replicate_collective_views(
          const CollectiveViewCreatorBase::RendezvousKey &key,
          std::map<LogicalRegion,
            CollectiveViewCreatorBase::CollectiveRendezvous> &rendezvous)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_manager != NULL);
      assert(!IS_COLLECTIVE(regions[key.region_index]));
      assert(!std::binary_search(check_collective_regions.begin(),
            check_collective_regions.end(), key.region_index));
#endif
      shard_manager->finalize_replicate_collective_views(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    InnerContext* SingleTask::create_implicit_context(void)
    //--------------------------------------------------------------------------
    {
      InnerContext *inner_ctx = new InnerContext(runtime, this, 
          get_depth(), false/*is inner*/, regions, output_regions,
          parent_req_indexes, virtual_mapped, ApEvent::NO_AP_EVENT,
          0/*did*/, false/*inline*/, true/*implicit*/);
      execution_context = inner_ctx;
      execution_context->add_base_gc_ref(SINGLE_TASK_REF);
      return inner_ctx;
    }

    //--------------------------------------------------------------------------
    void SingleTask::configure_execution_context(InnerContext *inner_ctx)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      inner_ctx->configure_context(mapper, task_priority);
    }

    //--------------------------------------------------------------------------
    void SingleTask::set_shard_manager(ShardManager *manager)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_manager == NULL);
#endif
      shard_manager = manager;
      shard_manager->add_base_gc_ref(SINGLE_TASK_REF);
    }

    //--------------------------------------------------------------------------
    void SingleTask::validate_target_processors(
                                 const std::vector<Processor> &processors) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!processors.empty());
#endif
      // Make sure that they are all on the same node and of the same kind
      const Processor &first = processors.front();
      const Processor::Kind kind = first.kind();
      const AddressSpace space = first.address_space();
      for (unsigned idx = 0; idx < processors.size(); idx++)
      {
        const Processor &proc = processors[idx];
        if (!proc.exists())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output. Mapper %s requested an illegal "
                        "NO_PROC for a target processor when mapping task %s "
                        "(ID %lld).", mapper->get_mapper_name(), 
                        get_task_name(), get_unique_id())
        else if (proc.kind() != kind)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output. Mapper %s requested processor "
                        IDFMT " which is of kind %s when mapping task %s "
                        "(ID %lld), but the target processor " IDFMT " has "
                        "kind %s. Only one kind of processor is permitted.",
                        mapper->get_mapper_name(), proc.id, 
                        Processor::get_kind_name(proc.kind()), get_task_name(),
                        get_unique_id(), this->target_proc.id, 
                        Processor::get_kind_name(kind))
        if (proc.address_space() != space)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output. Mapper %s requested processor "
                        IDFMT " which is in address space %d when mapping "
                        "task %s (ID %lld) but the target processor " IDFMT 
                        "is in address space %d. All target processors must "
                        "be in the same address space.", 
                        mapper->get_mapper_name(), proc.id,
                        proc.address_space(), get_task_name(), get_unique_id(), 
                        this->target_proc.id, space)
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::invoke_mapper(MustEpochOp *must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      Mapper::MapTaskInput input;
      Mapper::MapTaskOutput output;
      output.profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      output.copy_fill_priority = 0;
      // Initialize the mapping input which also does all the traversal
      // down to the target nodes
      initialize_map_task_input(input, output, must_epoch_owner); 
      // Now we can invoke the mapper to do the mapping
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_map_task(this, input, output);
      copy_fill_priority = output.copy_fill_priority;
      // Now we can convert the mapper output into our physical instances
      finalize_map_task_output(input, output, must_epoch_owner);
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert((remote_trace_recorder != NULL) ||
                ((tpl != NULL) && tpl->is_recording()));
        assert(futures.size() == future_memories.size());
#endif
        // We swapped this in finalize output so we need to restore it 
        // here if we're going to record it
        if (!futures.empty())
          output.future_locations = future_memories;
        const TraceLocalID tlid = get_trace_local_id();
        if (remote_trace_recorder != NULL)
          remote_trace_recorder->record_mapper_output(tlid, output,
              physical_instances, map_applied_conditions);
        else
          tpl->record_mapper_output(tlid, output, physical_instances,
              map_applied_conditions);
      }
    }

    //--------------------------------------------------------------------------
    bool SingleTask::replicate_task(void)
    //--------------------------------------------------------------------------
    {
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      // There are some local invariants checked here, but there are more
      // of them checked in select_task_options right after the mapper call
      // that decides whether we're going to try to replicate this task
      if (is_recording())
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_UNSUPPORTED_REPLICATION,
            "Unsupported request to replicate task %s (UID %lld) during "
            "trace capture by mapper %s. Legion does not currently support "
            "replication of tasks inside of physical traces at the moment. "
            "You can request support for this feature by emailing the "
            "the Legion developers list or opening a github issue. The "
            "mapper call to replicate_task is being elided.",
            get_task_name(), get_unique_id(), mapper->get_mapper_name())
        return false;
      }
      Mapper::ReplicateTaskInput input;
      Mapper::ReplicateTaskOutput output;
      output.chosen_variant = 0;
      mapper->invoke_replicate_task(this, input, output);
      // If we don't have more than one target processor then we're not
      // actually going to replicate this task
      if (output.target_processors.size() <= 1)
        return false;
      VariantImpl *var_impl = NULL;
      if (output.leaf_variants.empty())
      {
        var_impl = runtime->find_variant_impl(task_id,
            output.chosen_variant, true/*can_fail*/);
        if (var_impl == NULL)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper selected an invalid task "
                        "variant %d for task %s (UID %lld) that was chosen "
                        "to be replicated.", "replicate_task",
                        mapper->get_mapper_name(), output.chosen_variant,
                        get_task_name(), get_unique_id())
        // Check that the chosen variant is replicable
        if (!var_impl->is_replicable())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper failed to pick an valid task "
                        "variant %d for task %s (UID %lld) that was chosen "
                        "to be replicated. Task variants selected for "
                        "replication must be marked as replicable variants.",
                        "replicate_task", mapper->get_mapper_name(),
                        output.chosen_variant, get_task_name(), get_unique_id())
      }
      else
      {
        output.chosen_variant = 0;
        if (output.leaf_variants.size() != output.target_processors.size())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper provided %zd leaf variants "
                        "for %zd target processors for task %s (UID %lld). "
                        "The same number of leaf variants must be provided "
                        "as target processors.", "replicate_task", 
                        mapper->get_mapper_name(), output.leaf_variants.size(),
                        output.target_processors.size(), get_task_name(), 
                        get_unique_id())
        for (unsigned idx = 0; idx < output.leaf_variants.size(); idx++)
        {
          VariantImpl *impl = runtime->find_variant_impl(task_id,
            output.leaf_variants[idx], true/*can_fail*/);
          if (impl == NULL)
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of '%s' "
                          "on mapper %s. Mapper selected an invalid leaf task "
                          "variant %d for task %s (UID %lld) that was chosen "
                          "to be replicated.", "replicate_task",
                          mapper->get_mapper_name(), output.leaf_variants[idx],
                          get_task_name(), get_unique_id())
          if (var_impl == NULL)
            var_impl = impl;
          // Check that the chosen variant is a leaf
          if (!impl->is_leaf())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of '%s' "
                          "on mapper %s. Mapper failed to pick an valid task "
                          "variant %d for task %s (UID %lld) that was chosen "
                          "to be replicated. All variants provided in the "
                          "leaf_variants must be leaf task variants.",
                          "replicate_task", mapper->get_mapper_name(),
                          output.leaf_variants[idx], get_task_name(), 
                          get_unique_id())
          // Check that the chosen variant is replicable
          if (!impl->is_replicable())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                          "Invalid mapper output from invocation of '%s' "
                          "on mapper %s. Mapper failed to pick an valid task "
                          "variant %d for task %s (UID %lld) that was chosen "
                          "to be replicated. Task variants selected for "
                          "replication must be marked as replicable variants.",
                          "replicate_task", mapper->get_mapper_name(),
                          output.leaf_variants[idx], get_task_name(), 
                          get_unique_id()) 
        }
      }
      if (!runtime->unsafe_mapper)
      {
        // Check that all the processors exist
        for (unsigned idx = 0; idx < output.target_processors.size(); idx++)
          if (!output.target_processors[idx].exists())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper specified a NO_PROC in the "
                        "vector of target processors when replicating "
                        "task %s (UID %lld). All processors in "
                        "target_processors must exist.", "replicate_task",
                         mapper->get_mapper_name(),
                         get_task_name(), get_unique_id())
        // Check that the chosen variant works with all the targets processors
        if (output.leaf_variants.empty())
        {
          const ProcessorConstraint &constraint = 
            var_impl->execution_constraints.processor_constraint; 
          if (constraint.is_valid())
          {
            const char *proc_names[] = {
#define PROC_NAMES(name, desc) desc,
              REALM_PROCESSOR_KINDS(PROC_NAMES)
#undef PROC_NAMES
            };
            for (unsigned idx = 0; idx < output.target_processors.size(); idx++)
              if (!constraint.can_use(output.target_processors[idx].kind()))
                REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper specified %s processor " IDFMT 
                        " which cannot be used with variant %d when "
                        "replicating task %s (UID %lld) as the variant does "
                        "not support that kind of processor.", "replicate_task",
                         mapper->get_mapper_name(), 
                         proc_names[output.target_processors[idx].kind()],
                         output.target_processors[idx].id,
                         output.chosen_variant, 
                         get_task_name(), get_unique_id())
          }
        }
        else
        {
          const char *proc_names[] = {
#define PROC_NAMES(name, desc) desc,
            REALM_PROCESSOR_KINDS(PROC_NAMES)
#undef PROC_NAMES
          };
          for (unsigned idx = 0; idx < output.target_processors.size(); idx++)
          {
            VariantImpl *impl = runtime->find_variant_impl(task_id,
                output.leaf_variants[idx], false/*can_fail*/);
            const ProcessorConstraint &constraint = 
              impl->execution_constraints.processor_constraint; 
            if (constraint.is_valid() &&
                !constraint.can_use(output.target_processors[idx].kind()))
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper specified %s processor " IDFMT 
                        " which cannot be used with variant %d when "
                        "replicating task %s (UID %lld) as the variant does "
                        "not support that kind of processor.", "replicate_task",
                         mapper->get_mapper_name(), 
                         proc_names[output.target_processors[idx].kind()],
                         output.target_processors[idx].id,
                         output.leaf_variants[idx], 
                         get_task_name(), get_unique_id())
          }
        }
        // If the chosen variant is not a leaf check that processors are unique
        // Note that if the chosen variant is a leaf then they don't need to be
        // unique since the different shards won't need to synchronize
        if (!var_impl->is_leaf())
        {
          std::vector<Processor> sorted_procs = output.target_processors;
          std::sort(sorted_procs.begin(), sorted_procs.end());
          for (unsigned idx = 1; idx < sorted_procs.size(); idx++)
            if (sorted_procs[idx-1] == sorted_procs[idx])
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper provided duplicate target "
                        "processors for non-leaf task variant %d when "
                        "replicating task %s (UID %lld). In order to control "
                        "replicate a task all the target processors must be "
                        "unique.", "replicate_task", mapper->get_mapper_name(),
                        output.chosen_variant, get_task_name(), get_unique_id())
        }
        // Check that shard points match the size target processors if not empty
        if (!output.shard_points.empty())
        {
          if (!output.shard_domain.exists())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper provided shard_points without "
                        "providing an associated shard_domain when replicating "
                        "task %s (UID %lld). A shard domain must also be "
                        "provided in conjunction with a set of shard points.",
                        "replicate_task", mapper->get_mapper_name(),
                        get_task_name(), get_unique_id())
          if (output.shard_points.size() != output.target_processors.size())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper provided %zd shard_points "
                        "which does not match the %zd target processors "
                        "specified when replicating task %s (UID %lld). If "
                        "shard_points are provided they must exactly match the "
                        "number of target processors.", "replicate_task", 
                        mapper->get_mapper_name(), output.shard_points.size(),
                        output.target_processors.size(),
                        get_task_name(), get_unique_id())
          std::vector<DomainPoint> sorted_points = output.shard_points;
          std::sort(sorted_points.begin(), sorted_points.end());
          for (unsigned idx = 1; idx < sorted_points.size(); idx++)
            if (sorted_points[idx-1] == sorted_points[idx])
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper provided duplicate shard "
                        "points when replicating task %s (UID %lld). In "
                        "order to control replicate a task all the target "
                        "processors must be unique.", "replicate_task", 
                        mapper->get_mapper_name(),
                        get_task_name(), get_unique_id())
        }
        // Check that shard domain volume matches number of points if not empty
        if (output.shard_domain.exists())
        {
          if (output.shard_points.empty())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper provided shard_domain without "
                        "providing any associated shard_points when replicating"
                        " task %s (UID %lld). The shard_points data structure "
                        "must also be populated in conjunction with a "
                        "shard_domain.", "replicate_task",
                        mapper->get_mapper_name(),
                        get_task_name(), get_unique_id())
          if (output.shard_points.size() != output.shard_domain.get_volume())
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper provided %zd shard_points "
                        "for shard_domain with %zd points when replicating "
                        "task %s (UID %lld). The number of shard_points must "
                        "exactly match the volume of the shard_domain.",
                        "replicate_task", mapper->get_mapper_name(),
                        output.shard_points.size(), 
                        output.shard_domain.get_volume(),
                        get_task_name(), get_unique_id())
          for (unsigned idx = 0; idx < output.shard_points.size(); idx++)
            if (!output.shard_domain.contains(output.shard_points[idx]))
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of '%s' "
                        "on mapper %s. Mapper provided a point in shard_points "
                        "that is not contained in the shard_domain when "
                        "replicating task %s (UID %lld). Each point in "
                        "shard_points must exist in the shard_domain.", 
                        "replicate_task", mapper->get_mapper_name(),
                        get_task_name(), get_unique_id())
        }
      } 
      // Start building the data structures needed to make the ShardManager
      std::vector<DomainPoint> sorted_points;
      sorted_points.reserve(output.target_processors.size());
      std::vector<ShardID> shard_lookup;
      shard_lookup.reserve(output.target_processors.size());
      bool isomorphic_points = false;
      if (!output.shard_points.empty())
      {
        std::map<DomainPoint,ShardID> shard_mapping;
        const int dim = output.shard_points.front().get_dim();
        if (dim != 1)
          isomorphic_points = false;
        for (unsigned idx = 0; idx < output.shard_points.size(); idx++)
        {
          if (isomorphic_points && (output.shard_points[idx][0] != idx))
            isomorphic_points = false;
          if (output.shard_points[idx].get_dim() != dim)
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                "Mapper %s specified shard points with different "
                "dimensionalities of %d and %d for 'replicate_task' "
                "call for task %s (UID %lld). All shard points must have "
                "the same dimenstionality.", mapper->get_mapper_name(),
                dim, output.shard_points[idx].get_dim(),
                get_task_name(), get_unique_id())
          std::pair<std::map<DomainPoint,ShardID>::iterator,bool> result =
            shard_mapping.insert(std::pair<DomainPoint,ShardID>(
                  output.shard_points[idx],idx));
          if (!result.second)
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                "Mapper %s specified duplicate shard point names for shards "
                "%d and %d in 'replicate_task' mapper call for task %s "
                "(UID %lld). Each shard point must be given a unique name.",
                mapper->get_mapper_name(), result.first->second, idx,
                get_task_name(), get_unique_id())
        }
        for (std::map<DomainPoint,ShardID>::const_iterator it =
              shard_mapping.begin(); it != shard_mapping.end(); it++)
        {
          sorted_points.push_back(it->first);
          shard_lookup.push_back(it->second);
        }
        const int domain_dim = output.shard_domain.get_dim();
        if ((domain_dim > 0) && (domain_dim != dim))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
              "Mapper %s specified a 'shard_domain' output with "
              "dimensionality %d different than the %d dimension points "
              "in 'shard_points' in 'replicate_task' call for task %s "
              "(UID %lld). The dimensionality of 'shard_domain' must "
              "match the dimensionality of the 'shard_points'.",
              mapper->get_mapper_name(), domain_dim, dim,
              get_task_name(), get_unique_id())
      }
      else
      {
        // Mapper didn't specify it so we can fill it in
        output.shard_domain = Domain(DomainPoint(0),
            DomainPoint(output.target_processors.size()-1));
        output.shard_points.reserve(output.target_processors.size());
        for (unsigned idx = 0; idx < output.target_processors.size(); idx++)
        {
          output.shard_points.push_back(DomainPoint(idx));
          sorted_points.push_back(DomainPoint(idx));
          shard_lookup.push_back(idx);
        }
      }
      // Construct the collective mapping
      std::vector<AddressSpaceID> spaces(output.target_processors.size()+1);
      for (unsigned idx = 0; idx < output.target_processors.size(); idx++)
        spaces[idx] = output.target_processors[idx].address_space();
      // Make sure we include our local space too
      spaces.back() = runtime->address_space;
      std::sort(spaces.begin(), spaces.end());
      // Uniquify them
      std::vector<AddressSpaceID>::iterator last = 
        std::unique(spaces.begin(), spaces.end());
      spaces.erase(last, spaces.end());
      // The shard manager will take ownership of this
      CollectiveMapping *mapping =
        new CollectiveMapping(spaces, runtime->legion_collective_radix);
      const DistributedID manager_did = runtime->get_available_distributed_id();
      if (runtime->legion_spy_enabled)
        LegionSpy::log_replication(get_unique_id(), manager_did,
                                   !var_impl->is_leaf());
#ifdef DEBUG_LEGION
      assert(shard_manager == NULL);
#endif
      std::vector<ShardID> local_shards;
      for (ShardID idx = 0; idx < output.target_processors.size(); idx++)
      {
        const Processor processor = output.target_processors[idx];
        if (processor.address_space() != runtime->address_space)
          continue;
        local_shards.push_back(idx);
      }
      shard_manager = new ShardManager(runtime, manager_did, mapping,
          local_shards.size(), is_top_level_task(), isomorphic_points,
          !var_impl->is_leaf(), output.shard_domain, 
          std::move(output.shard_points), std::move(sorted_points),
          std::move(shard_lookup), this);
      shard_manager->add_base_gc_ref(SINGLE_TASK_REF);
      // Now create our local shards and start them mapping
      for (unsigned idx = 0; idx < local_shards.size(); idx++)
        shard_manager->create_shard(local_shards[idx], 
            output.target_processors[local_shards[idx]],
            output.leaf_variants.empty() ? output.chosen_variant :
              output.leaf_variants[local_shards[idx]], parent_ctx, this);
      // Distribute the shard manager and launch the shards 
      shard_manager->distribute_explicit(this, output.chosen_variant,
          output.target_processors, output.leaf_variants);
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent SingleTask::map_all_regions(MustEpochOp *must_epoch_op,
                                        const DeferMappingArgs *defer_args)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, MAP_ALL_REGIONS_CALL);
      // Check to see if we need to make a remote trace recorder
      if (is_remote() && is_recording() && (remote_trace_recorder == NULL))
      {
        const RtUserEvent remote_applied = Runtime::create_rt_user_event();
        remote_trace_recorder = new RemoteTraceRecorder(runtime,
            orig_proc.address_space(), runtime->address_space, 
            get_trace_local_id(), tpl, remote_applied);
        remote_trace_recorder->add_recorder_reference();
        map_applied_conditions.insert(remote_applied);
#ifdef DEBUG_LEGION
        assert(!single_task_termination.exists());
#endif
        // Really unusual case here, if we're going to be doing remote tracing
        // then we need to get an event from the owner node because some kinds
        // of tracing (e.g. those with control replication) don't work otherwise
        remote_trace_recorder->request_term_event(single_task_termination);
      }
      // Create our task termination event at this point
      // Note that tracing doesn't track this as a user event, it is just
      // a name we're making for the termination event
      if (!single_task_termination.exists())
      {
        single_task_termination = Runtime::create_ap_user_event(NULL); 
        record_completion_effect(single_task_termination);
      }
      // Only do this the first or second time through
      if ((defer_args == NULL) || (defer_args->invocation_count < 2))
      {
        if (request_valid_instances)
        {
          // If the mapper wants valid instances we first need to do our
          // versioning analysis and then call the mapper
          if ((defer_args == NULL/*first invocation*/) ||
              (defer_args->invocation_count == 0))
          {
            const RtEvent version_ready_event = 
              perform_versioning_analysis(false/*post mapper*/);
            if (version_ready_event.exists() && 
                !version_ready_event.has_triggered())
            return defer_perform_mapping(version_ready_event, must_epoch_op,
                                         defer_args, 1/*invocation count*/);
          }
          // Now do the mapping call
          invoke_mapper(must_epoch_op);
        }
        else
        {
          // If the mapper doesn't need valid instances, we do the mapper
          // call first and then see if we need to do any versioning analysis
          if ((defer_args == NULL/*first invocation*/) ||
              (defer_args->invocation_count == 0))
          {
            invoke_mapper(must_epoch_op);
            const RtEvent version_ready_event = 
              perform_versioning_analysis(true/*post mapper*/);
            if (version_ready_event.exists() && 
                !version_ready_event.has_triggered())
            return defer_perform_mapping(version_ready_event, must_epoch_op,
                                         defer_args, 1/*invocation count*/);
          }
        }
      }
      // If we have any intra-space mapping dependences that haven't triggered
      // then we need to defer ourselves until they have occurred
      if (!intra_space_mapping_dependences.empty())
      {
        const RtEvent ready = 
          Runtime::merge_events(intra_space_mapping_dependences);
        intra_space_mapping_dependences.clear();
        if (ready.exists() && !ready.has_triggered())
          return defer_perform_mapping(ready, must_epoch_op,
                                       defer_args, 2/*invocation count*/);
      } 
      // See if we have a remote trace info to use, if we don't then make
      // our trace info and do the initialization
      const TraceInfo trace_info = is_remote() ?
        TraceInfo(this, remote_trace_recorder) : TraceInfo(this);
      // If we'r recording then record the replay map task
      if (is_recording())
        trace_info.record_replay_mapping(single_task_termination,
            TASK_OP_KIND, (get_task_kind() != INDIVIDUAL_TASK_KIND));
      ApEvent init_precondition = compute_sync_precondition(trace_info);
      // After we've got our results, apply the state to the region tree
      size_t region_count = get_region_count();
      region_preconditions.resize(region_count);
      if (region_count > 0)
      {
        if (regions.size() == 1 && output_regions.empty())
        {
          if (!no_access_regions[0] && !virtual_mapped[0])
          {
            const bool record_valid = !std::binary_search(
                untracked_valid_regions.begin(),
                untracked_valid_regions.end(), 0);
            const bool check_collective = 
              IS_COLLECTIVE(regions.front()) || std::binary_search(
                check_collective_regions.begin(),
                check_collective_regions.end(), 0);
            region_preconditions.back() =
              runtime->forest->physical_perform_updates_and_registration(
                  regions[0], version_infos[0], this, 0, 
                  init_precondition, single_task_termination,
                  physical_instances[0], source_instances[0],
                  PhysicalTraceInfo(trace_info, 0), map_applied_conditions,
#ifdef DEBUG_LEGION
                                        get_logging_name(),
                                        unique_op_id,
#endif
                                        check_collective,
                                        record_valid);
#ifdef DEBUG_LEGION
            dump_physical_state(&regions[0], 0);
#endif
          }
        }
        else
        {
          unsigned read_only_count = 0;
          std::vector<unsigned> performed_regions;
          performed_regions.reserve(region_count);
          std::vector<UpdateAnalysis*> analyses(region_count, NULL);
          std::vector<RtEvent> reg_pre(region_count, RtEvent::NO_RT_EVENT);
          for (unsigned idx = 0; idx < logical_regions.size(); idx++)
          {
            if (no_access_regions[idx])
              continue;
            VersionInfo &local_info = get_version_info(idx);
            // If we virtual mapped it, there is nothing to do
            if (virtual_mapped[idx])
              continue;
            performed_regions.push_back(idx);
            const bool record_valid = !std::binary_search(
                untracked_valid_regions.begin(),
                untracked_valid_regions.end(), idx);
            const bool check_collective = 
              IS_COLLECTIVE(logical_regions[idx]) || std::binary_search(
                check_collective_regions.begin(),
                check_collective_regions.end(), idx);
            // apply the results of the mapping to the tree
            reg_pre[idx] = runtime->forest->physical_perform_updates(
                                        logical_regions[idx], local_info,
                                        this, idx, init_precondition,
                                        single_task_termination,
                                        physical_instances[idx],
                                        source_instances[idx],
                                        PhysicalTraceInfo(trace_info, idx),
                                        map_applied_conditions,
                                        analyses[idx],
#ifdef DEBUG_LEGION
                                        get_logging_name(),
                                        unique_op_id,
#endif
                                        check_collective,
                                        record_valid);
            if (IS_READ_ONLY(logical_regions[idx]))
              read_only_count++;
          }
          // In order to avoid cycles when mapping multiple tasks in parallel
          // with read-only requirements, we need to guarantee that all read-only
          // copies are issued before we can perform any registrations for the
          // task that will be using their results.
          if (read_only_count > 1)
          {
            std::vector<RtEvent> read_only_preconditions;
            read_only_preconditions.reserve(read_only_count);
            std::vector<unsigned> read_only_regions;
            read_only_regions.reserve(read_only_count);
            for (std::vector<unsigned>::const_iterator it =
                  performed_regions.begin(); it !=
                  performed_regions.end(); it++)
            {
              if (!IS_READ_ONLY(logical_regions[*it]))
                continue;
              read_only_regions.push_back(*it);
              const RtEvent precondition = reg_pre[*it];
              if (precondition.exists())
                read_only_preconditions.push_back(precondition);
            }
            if (!read_only_preconditions.empty())
            {
              const RtEvent read_only_precondition =
                Runtime::merge_events(read_only_preconditions);
              if (read_only_precondition.exists())
              {
                for (std::vector<unsigned>::const_iterator it =
                      read_only_regions.begin(); it !=
                      read_only_regions.end(); it++)
                  reg_pre[*it] = read_only_precondition;
              }
            }
          }
          for (std::vector<unsigned>::const_iterator it = 
               performed_regions.begin(); it != performed_regions.end(); it++)
          {
            region_preconditions[*it] = 
              runtime->forest->physical_perform_registration(reg_pre[*it],
                                    analyses[*it], 
                                    map_applied_conditions,
                                    logical_regions.is_output_created(*it));
#ifdef DEBUG_LEGION
            RegionRequirement *req = &logical_regions[*it];
            dump_physical_state(req, *it);
#endif
          }
        }
        if (perform_postmap)
          perform_post_mapping(trace_info);
      } // if (!regions.empty())
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert(output_regions.empty());
#endif
        const TraceInfo trace_info = is_remote() ?
          TraceInfo(this, remote_trace_recorder) : TraceInfo(this);
        if (execution_fence_event.exists())
          region_preconditions.push_back(execution_fence_event);
        ApEvent ready_event = 
          Runtime::merge_events(&trace_info, region_preconditions);
        if (execution_fence_event.exists())
          region_preconditions.pop_back();
        const TraceLocalID tlid = get_trace_local_id();
        if (!atomic_locks.empty())
          trace_info.record_reservations(tlid, atomic_locks,
                                         map_applied_conditions);
        // Record the replay completion once we know we have all the effects
        RtEvent record_replay_precondition;
        if (!map_applied_conditions.empty())
        {
          // If we have a remote trace recorder, make sure we don't
          // accidentally include ourselves in the preconditions for
          // ourself which will cause a recording deadlock
          if (remote_trace_recorder != NULL)
            map_applied_conditions.erase(remote_trace_recorder->applied_event);
          record_replay_precondition =
            Runtime::merge_events(map_applied_conditions);
          map_applied_conditions.clear();
        }
        const RtEvent replay_recorded = record_complete_replay(trace_info,
                                  record_replay_precondition, ready_event);
        if (replay_recorded.exists())
          map_applied_conditions.insert(replay_recorded);
      }
      if (remote_trace_recorder != NULL)
      {
        if (remote_trace_recorder->remove_recorder_reference())
          delete remote_trace_recorder;
        remote_trace_recorder = NULL;
      }
      if (must_epoch_op != NULL)
      {
        // If we are part of a must epoch operation, then report the 
        // event that describes when all of our mapping activies are done
        RtEvent mapping_applied;
        if (!map_applied_conditions.empty())
          mapping_applied = Runtime::merge_events(map_applied_conditions);
        must_epoch_op->record_mapped_event(index_point, mapping_applied);
      }
      // If we're a leaf task then call handle post mapped now since we
      // know we're not going to get it from the context
      if (is_leaf())
        handle_post_mapped(RtEvent::NO_RT_EVENT);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_post_mapping(const TraceInfo &trace_info)
    //--------------------------------------------------------------------------
    {
      Mapper::PostMapInput input;
      Mapper::PostMapOutput output;
      input.mapped_regions.resize(regions.size());
      input.valid_instances.resize(regions.size());
      input.valid_collectives.resize(regions.size());
      output.chosen_instances.resize(regions.size());
      output.source_instances.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (no_access_regions[idx] || virtual_mapped[idx])
          continue;
        // Don't need to actually traverse very far, but we do need the
        // valid instances for all the regions
        if (request_valid_instances)
        {
          InstanceSet postmap_valid;
          FieldMaskSet<ReplicatedView> collectives;
          runtime->forest->physical_premap_region(this, idx, regions[idx], 
                                                  get_version_info(idx),
                                                  postmap_valid, collectives,
                                                  map_applied_conditions);
          // No need to filter these because they are on the way out
          prepare_for_mapping(postmap_valid, collectives,
              input.valid_instances[idx], input.valid_collectives[idx]);  
        }
        FieldMaskSet<ReplicatedView> no_collectives;
        prepare_for_mapping(physical_instances[idx], no_collectives,
            input.mapped_regions[idx], input.valid_collectives[idx]);
      }
      // Now we can do the mapper call
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_post_map_task(this, input, output);
      // Check and register the results
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (no_access_regions[idx] || virtual_mapped[idx])
          continue;
        if (output.chosen_instances[idx].empty())
          continue;
        RegionRequirement &req = regions[idx];
        if (req.is_restricted())
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_POST,
                          "Mapper %s requested post mapping "
                          "instances be created for region requirement %d "
                          "of task %s (ID %lld), but this region requirement "
                          "is restricted. The request is being ignored.",
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
          continue;
        }
        if (IS_NO_ACCESS(req))
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_POST,
                          "Mapper %s requested post mapping "
                          "instances be created for region requirement %d "
                          "of task %s (ID %lld), but this region requirement "
                          "has NO_ACCESS privileges. The request is being "
                          "ignored.", mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
          continue;
        }
        if (IS_REDUCE(req))
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_POST,
                          "Mapper %s requested post mapping "
                          "instances be created for region requirement %d "
                          "of task %s (ID %lld), but this region requirement "
                          "has REDUCE privileges. The request is being "
                          "ignored.", mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
          continue;
        }
        // Convert the post-mapping  
        InstanceSet result;
        RegionTreeID bad_tree = 0;
        std::vector<PhysicalManager*> unacquired;
        bool had_composite = 
          runtime->forest->physical_convert_postmapping(this, req,
                              output.chosen_instances[idx], result, bad_tree,
                              runtime->unsafe_mapper ? NULL : 
                                get_acquired_instances_ref(),
                              unacquired, !runtime->unsafe_mapper);
        if (bad_tree > 0)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from 'postmap_task' invocation "
                        "on mapper %s. Mapper provided an instance from region "
                        "tree %d for use in satisfying region requirement %d "
                        "of task %s (ID %lld) whose region is from region tree "
                        "%d.", mapper->get_mapper_name(), bad_tree,
                        idx, get_task_name(), get_unique_id(), 
                        regions[idx].region.get_tree_id())
        if (!unacquired.empty())
        {
          std::map<PhysicalManager*,unsigned> *acquired_instances = 
            get_acquired_instances_ref();
          for (std::vector<PhysicalManager*>::const_iterator uit = 
                unacquired.begin(); uit != unacquired.end(); uit++)
          {
            if (acquired_instances->find(*uit) == acquired_instances->end())
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from 'postmap_task' "
                            "invocation on mapper %s. Mapper selected "
                            "physical instance for region requirement "
                            "%d of task %s (ID %lld) which has already "
                            "been collected. If the mapper had properly "
                            "acquired this instance as part of the mapper "
                            "call it would have detected this. Please "
                            "update the mapper to abide by proper mapping "
                            "conventions.", mapper->get_mapper_name(),
                            idx, get_task_name(), get_unique_id())
          }
          // If we did successfully acquire them, still issue the warning
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_FAILED_ACQUIRE,
                          "mapper %s failed to acquires instances "
                          "for region requirement %d of task %s (ID %lld) "
                          "in 'postmap_task' call. You may experience "
                          "undefined behavior as a consequence.",
                          mapper->get_mapper_name(), idx, 
                          get_task_name(), get_unique_id());
        }
        if (had_composite)
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_COMPOSITE,
                          "Mapper %s requested a composite "
                          "instance be created for region requirement %d "
                          "of task %s (ID %lld) for a post mapping. The "
                          "request is being ignored.",
                          mapper->get_mapper_name(), idx,
                          get_task_name(), get_unique_id());
          continue;
        }
        if (!runtime->unsafe_mapper)
        {
          std::vector<LogicalRegion> regions_to_check(1, 
                                        regions[idx].region);
          for (unsigned check_idx = 0; check_idx < result.size(); check_idx++)
          {
            PhysicalManager *manager = result[check_idx].get_physical_manager();
            if (!manager->meets_regions(regions_to_check))
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                            "Invalid mapper output from invocation of "
                            "'postmap_task' on mapper %s. Mapper specified an "
                            "instance region requirement %d of task %s "
                            "(ID %lld) that does not meet the logical region "
                            "requirement.", mapper->get_mapper_name(), idx, 
                            get_task_name(), get_unique_id())
          }
        }
        log_mapping_decision(idx, regions[idx], result, true/*postmapping*/);
        // TODO: Implement physical tracing for postmapped regions
        if (is_memoizing())
          assert(false);
        // Register this with a no-event so that the instance can
        // be used as soon as it is valid from the copy to it
        // We also use read-only privileges to ensure that it doesn't
        // invalidate the other valid instances
        const PrivilegeMode mode = regions[idx].privilege;
        regions[idx].privilege = LEGION_READ_ONLY; 
        VersionInfo &local_version_info = get_version_info(idx);
        std::vector<PhysicalManager*> sources;
        if (!output.source_instances[idx].empty())
          runtime->forest->physical_convert_sources(this, regions[idx],
              output.source_instances[idx], sources, 
              !runtime->unsafe_mapper ? get_acquired_instances_ref() : NULL);
        runtime->forest->physical_perform_updates_and_registration(
                          regions[idx], local_version_info, this, idx,
                          single_task_termination/*wait for task to be done*/,
                          ApEvent::NO_AP_EVENT/*done immediately*/, 
                          result, sources, PhysicalTraceInfo(trace_info, idx), 
                          map_applied_conditions,
#ifdef DEBUG_LEGION
                          get_logging_name(), unique_op_id,
#endif
                          false/*check for collectives*/,
                          false/*track effects*/);
        regions[idx].privilege = mode; 
      }
    } 

    //--------------------------------------------------------------------------
    void SingleTask::check_future_return_bounds(FutureInstance *instance) const
    //--------------------------------------------------------------------------
    {
      VariantImpl *var_impl = 
        runtime->find_variant_impl(task_id, selected_variant);
      if (var_impl->has_return_type_size &&
          (var_impl->return_type_size < instance->size))
      {
        Provenance *provenance = get_provenance();
        if (provenance != NULL)
          REPORT_LEGION_ERROR(ERROR_FUTURE_SIZE_BOUNDS_EXCEEDED,
              "Task %s (UID %lld, provenance: %s) used a task "
              "variant with a maximum return size of %zd but "
              "returned a result of %zd bytes.",
              get_task_name(), get_unique_id(), provenance->human_str(),
              var_impl->return_type_size, instance->size)
        else
          REPORT_LEGION_ERROR(ERROR_FUTURE_SIZE_BOUNDS_EXCEEDED,
              "Task %s (UID %lld) used a task variant with a maximum "
              "return size of %zd but returned a result of %zd bytes.",
              get_task_name(), get_unique_id(),
              var_impl->return_type_size, instance->size)
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, LAUNCH_TASK_CALL);
#ifdef DEBUG_LEGION
      assert(logical_regions.size() == physical_instances.size());
      assert(logical_regions.size() == no_access_regions.size());
#endif 
      // If we haven't computed our virtual mapping information
      // yet (e.g. because we origin mapped) then we have to
      // do that now
      if (virtual_mapped.size() != regions.size())
      {
        virtual_mapped.resize(regions.size());
        for (unsigned idx = 0; idx < regions.size(); idx++)
          virtual_mapped[idx] = physical_instances[idx].is_virtual_mapping();
      }
      VariantImpl *variant = 
        runtime->find_variant_impl(task_id, selected_variant);
      // STEP 1: Compute the precondition for the task launch
      std::set<ApEvent> wait_on_events;
      if (execution_fence_event.exists())
        wait_on_events.insert(execution_fence_event);
      if (concurrent_fence_event.exists())
        wait_on_events.insert(concurrent_fence_event);
#ifdef LEGION_SPY
      // TODO: teach legion spy how to check the inner task optimization
      // for now we'll just turn it off whenever we are going to be
      // validating the runtime analysis
      const bool do_inner_task_optimization = false;
#else
      const bool do_inner_task_optimization = variant->is_inner();
#endif
      // Get the event to wait on unless we are 
      // doing the inner task optimization
      if (!do_inner_task_optimization)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
          if (region_preconditions[idx].exists())
            wait_on_events.insert(region_preconditions[idx]);
        for (unsigned idx = 0; idx < futures.size(); idx++)
        {
          FutureImpl *impl = futures[idx].impl; 
          if (impl == NULL)
            continue;
          ApEvent ready;
          if (idx < future_memories.size())
          {
            if (future_memories[idx].exists())
              ready = impl->find_application_instance_ready(
                                  future_memories[idx], this);
            else // skip requesting any futures mapped to NO_MEMORY
              continue;
          }
          else
            ready = impl->find_application_instance_ready(
                      runtime->runtime_system_memory, this);
          if (ready.exists())
            wait_on_events.insert(ready);
        }
      }
      // Now add get all the other preconditions for the launch
      for (unsigned idx = 0; idx < grants.size(); idx++)
      {
        GrantImpl *impl = grants[idx].impl;
        wait_on_events.insert(impl->acquire_grant());
      }
      for (unsigned idx = 0; idx < wait_barriers.size(); idx++)
      {
	ApEvent e = 
          Runtime::get_previous_phase(wait_barriers[idx].phase_barrier);
        wait_on_events.insert(e);
      }

      // STEP 2: Set up the task's context
      {
        const bool is_leaf_variant = variant->is_leaf();
        execution_context = create_execution_context(variant, 
            wait_on_events, inline_task, is_leaf_variant);
        std::vector<ApUserEvent> unmap_events(regions.size());
        std::vector<RegionRequirement> clone_requirements(regions.size());
        // Make physical regions for each our region requirements
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
#ifdef DEBUG_LEGION
          assert(regions[idx].handle_type == LEGION_SINGULAR_PROJECTION);
#endif
          // If it was virtual mapper so it doesn't matter anyway.
          if (virtual_mapped[idx] || no_access_regions[idx])
          {
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            execution_context->add_physical_region(clone_requirements[idx],
                false/*mapped*/, map_id, tag, unmap_events[idx],
                true/*virtual mapped*/, physical_instances[idx]);
          }
          else if (do_inner_task_optimization)
          {
            // If this is an inner task then we don't map
            // the region with a physical region, but instead
            // we mark that the unmap event which marks when
            // the region can be used by child tasks should
            // be the ready event.
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            // Also make the region requirement read-write to force
            // people to wait on the value
            if (!IS_REDUCE(regions[idx]))
              clone_requirements[idx].privilege = LEGION_READ_WRITE;
            execution_context->add_physical_region(clone_requirements[idx],
                false/*mapped*/, map_id, tag, unmap_events[idx],
                false/*virtual mapped*/, physical_instances[idx]);
#ifdef DEBUG_LEGION
            assert(unmap_events[idx].exists());
#endif
            // Trigger the user event when the region is 
            // actually ready to be used
            Runtime::trigger_event(NULL, unmap_events[idx],
                                   region_preconditions[idx]);
          }
          else
          { 
            // If this is not virtual mapped, here is where we
            // switch coherence modes from whatever they are in
            // the enclosing context to exclusive within the
            // context of this task
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            execution_context->add_physical_region(clone_requirements[idx],
                true/*mapped*/, map_id, tag, unmap_events[idx],
                false/*virtual mapped*/, physical_instances[idx]);
            // We reset the reference below after we've
            // initialized the local contexts and received
            // back the local instance references
          }
        }
        // Initialize output regions
        for (unsigned idx = 0; idx < output_regions.size(); ++idx)
          execution_context->add_output_region(output_regions[idx],
              physical_instances[regions.size() + idx],
              is_output_global(idx), is_output_valid(idx));

        // Initialize any region tree contexts
        execution_context->initialize_region_tree_contexts(clone_requirements,
                                                  version_infos, unmap_events);
        // Update the physical regions with any padding they might have
        if (variant->needs_padding)
          execution_context->record_padded_fields(variant);
      }
      // If we have a predicate event then merge that in here as well
      if (true_guard.exists())
        wait_on_events.insert(ApEvent(true_guard));
      // Merge together all the events for the start condition 
      ApEvent start_condition = Runtime::merge_events(NULL, wait_on_events);
      // If we're performing a concurrent index space task launch then we
      // need to perform an extra step here to ensure a global ordering 
      // between concurrent index space task launches on the same processor
      if (is_concurrent())
      {
#ifdef DEBUG_LEGION
        assert(target_processors.size() == 1);
#endif
        const OrderConcurrentLaunchArgs args(this, target_processors.front(),
            start_condition, variant->is_concurrent() ? selected_variant : 0);
        // Give this very high priority as it is likely on the critical path
        runtime->issue_runtime_meta_task(args, LG_RESOURCE_PRIORITY,
            Runtime::protect_event(start_condition));
        start_condition = args.ready;
      }
      // Need a copy of any locks to release on the stack since the 
      // atomic_locks cannot be touched after we launch the task
      std::vector<Reservation> to_release;
      if (!atomic_locks.empty())
      {
        // Take all the locks in order in the proper way
        to_release.reserve(atomic_locks.size());
        for (std::map<Reservation,bool>::const_iterator it = 
              atomic_locks.begin(); it != atomic_locks.end(); it++)
        {
          start_condition = Runtime::acquire_ap_reservation(it->first, 
                                          it->second, start_condition);
          to_release.push_back(it->first);
        }
      }
      // STEP 3: Finally we get to launch the task
      // Mark that we have an outstanding task in this context 
      if (inline_task)
        parent_ctx->increment_inlined();
      else
        parent_ctx->increment_pending();
      // Note there is a potential scary race condition to be aware of here: 
      // once we launch this task it's possible for this task to run and 
      // clean up before we finish the execution of this function thereby
      // invalidating this SingleTask object's fields.  This means
      // that we need to save any variables we need for after the task
      // launch here on the stack before they can be invalidated.
#ifdef DEBUG_LEGION
      assert(!target_processors.empty());
#endif
      Processor launch_processor = target_processors[0];
      if (target_processors.size() > 1)
      {
        // Find the processor group for all the target processors
        launch_processor = runtime->find_processor_group(target_processors);
      }
      Realm::ProfilingRequestSet profiling_requests;
      // If the mapper requested profiling add that now too
      if (!task_profiling_requests.empty())
      {
        // See if we have any realm requests
        std::set<Realm::ProfilingMeasurementID> realm_measurements;
        for (std::vector<ProfilingMeasurementID>::const_iterator it = 
              task_profiling_requests.begin(); it != 
              task_profiling_requests.end(); it++)
        {
          if ((*it) < Mapping::PMID_LEGION_FIRST)
            realm_measurements.insert((Realm::ProfilingMeasurementID)(*it));
          else if ((*it) == Mapping::PMID_RUNTIME_OVERHEAD)
            execution_context->initialize_overhead_profiler();
          else
            assert(false); // should never get here
        }
        if (!realm_measurements.empty())
        {
          OpProfilingResponse response(this, 0, 0, false/*fill*/, true/*task*/);
          Realm::ProfilingRequest &request = profiling_requests.add_request(
              runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
              &response, sizeof(response));
          request.add_measurements(realm_measurements);
          // No need to increment the number of outstanding profiling
          // requests here since it was already done when we invoked
          // the mapper (see SingleTask::invoke_mapper)
          // The exeception is for origin-mapped remote tasks on which
          // we're going to need to send a message back to the owner
          if (is_remote() && is_origin_mapped())
          {
#ifdef DEBUG_LEGION
            assert(outstanding_profiling_requests.load() == 0);
            assert(!profiling_reported.exists());
#endif
            outstanding_profiling_requests.store(1);
            profiling_reported = Runtime::create_rt_user_event();
          }
        }
      }
      // Make a RtEvent copy of the false_guard in the case that we
      // are going to execute this task with a predicate and we'll
      // need to launch the misspeculation task after we launch the 
      // actual task itself. We have to pull this onto the stack before 
      // launching the task itself as the task might ultimately be cleaned
      // up before we're done executing this function so we can't touch 
      // any member variables after we launch it
      const RtEvent misspeculation_precondition = RtEvent(false_guard);
      if (runtime->legion_spy_enabled)
      {
        LegionSpy::log_variant_decision(unique_op_id, selected_variant);
#ifdef LEGION_SPY
        LegionSpy::log_operation_events(unique_op_id, start_condition,
                                        single_task_termination);
#endif
        LegionSpy::log_task_priority(unique_op_id, task_priority);
        for (unsigned idx = 0; idx < futures.size(); idx++)
        {
          FutureImpl *impl = futures[idx].impl;
          LegionSpy::log_future_use(unique_op_id, impl->did);
        }
      }
      // If this is a leaf task variant, then we can immediately trigger
      // the single_task_termination event dependent on the task_launch_event
      // because we know there will be no child operations we need to wait for
      // We have to pull it onto the stack here though to avoid the race 
      // condition with us getting pre-empted and the task running to completion
      // before we get a chance to trigger the event
      ApUserEvent chain_task_termination;
      if (variant->is_leaf() && !inline_task)
      {
#ifdef DEBUG_LEGION
        assert(single_task_termination.exists());
#endif
        chain_task_termination = single_task_termination;
      }
      ApEvent task_launch_event = variant->dispatch_task(launch_processor, this,
         execution_context, start_condition, task_priority, profiling_requests);
      // Release any reservations that we took on behalf of this task
      // Note this happens before protection of the event for predication
      // because the acquires were also subject to poisoning so we either
      // want all the releases to be done or poisoned the same as the acquires
      if (!to_release.empty())
      {
        for (std::vector<Reservation>::const_iterator it = 
              to_release.begin(); it != to_release.end(); it++)
          Runtime::release_reservation(*it, task_launch_event);
      }
      // If this task was predicated then we need to protect everything that
      // comes after this from the predication poison
      if (true_guard.exists())
      {
        task_launch_event = Runtime::ignorefaults(task_launch_event);
        // Also merge in the original preconditions so that is reflected 
        // downstream in the event chain still for things like postconditions
        // Make sure to prune out the true guard that we added here
        wait_on_events.erase(ApEvent(true_guard));
        if (!wait_on_events.empty())
        {
          wait_on_events.insert(task_launch_event);
          task_launch_event = Runtime::merge_events(NULL, wait_on_events);
        }
      }
      if (chain_task_termination.exists())
        Runtime::trigger_event(NULL, chain_task_termination, task_launch_event);
      // Finally if this is a predicated task and we have a speculative
      // guard then we need to launch a meta task to handle the case
      // where the task misspeculates
      if (misspeculation_precondition.exists())
      {
        MispredicationTaskArgs args(this);
        // Make sure this runs on an application processor where the
        // original task was going to go 
        runtime->issue_runtime_meta_task(args, LG_LATENCY_WORK_PRIORITY, 
                                         misspeculation_precondition);
        // Fun little trick here: decrement the outstanding meta-task
        // counts for the mis-speculation task in case it doesn't run
        // If it does run, we'll increment the counts again
#ifdef DEBUG_LEGION
        runtime->decrement_total_outstanding_tasks(
            MispredicationTaskArgs::TASK_ID, true/*meta*/);
#else
        runtime->decrement_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
        runtime->outstanding_counts[
          MispredicationTaskArgs::TASK_ID].fetch_sub(1);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_profiling_requests(Serializer &rez,
                                             std::set<RtEvent> &applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(copy_fill_priority);
      rez.serialize<size_t>(copy_profiling_requests.size());
      if (!copy_profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < copy_profiling_requests.size(); idx++)
          rez.serialize(copy_profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Send a message to the owner with an update for the extra counts
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        rez.serialize<RtEvent>(done_event);
        applied.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    int SingleTask::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any copy profiling requests
      if (copy_profiling_requests.empty())
        return copy_fill_priority;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request(
        runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
        &response, sizeof(response));
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            copy_profiling_requests.begin(); it != 
            copy_profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
      return copy_fill_priority;
    }

    //--------------------------------------------------------------------------
    void SingleTask::handle_profiling_response(
                                       const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id); 
      const OpProfilingResponse *task_prof = 
            static_cast<const OpProfilingResponse*>(base);
      // First see if this is a task response for an origin-mapped task
      // on a remote node that needs to be sent back to the origin node
      if (task_prof->task && is_origin_mapped() && is_remote())
      {
        // We need to send this response back to the owner node along
        // with the overhead tracker
        SingleTask *orig_task = get_origin_task();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(orig_task);
          rez.serialize(orig_length);
          rez.serialize(orig, orig_length);
          if (execution_context->overhead_profiler)
          {
            rez.serialize<bool>(true);
            // Only pack the bits that we need for the profiling response
            rez.serialize((const void*)execution_context->overhead_profiler,
                sizeof(Mapping::ProfilingMeasurements::RuntimeOverhead));
          }
          else
            rez.serialize<bool>(false);
        }
        runtime->send_remote_task_profiling_response(orig_proc, rez);
      }
      else
      {
        // Check to see if we are done mapping, if not then we need to defer
        // this until we are done mapping so we know how many
        if (!mapped_event.has_triggered())
        {
          // Take the lock and see if we lost the race
          AutoLock o_lock(op_lock);
          if (!mapped_event.has_triggered())
          {
            // Save this profiling response for later until we know the
            // full count of profiling responses
            profiling_info.resize(profiling_info.size() + 1);
            SingleProfilingInfo &info = profiling_info.back();
            info.task_response = task_prof->task; 
            info.region_requirement_index = task_prof->src;
            info.fill_response = task_prof->fill;
            info.buffer_size = orig_length;
            info.buffer = malloc(orig_length);
            memcpy(info.buffer, orig, orig_length);
            if (info.task_response)
            {
              // If we had an overhead profiler
              // see if this is the callback for the task
              if (execution_context->overhead_profiler != NULL)
                // This is the callback for the task itself
                info.profiling_responses.attach_overhead(
                    execution_context->overhead_profiler);
            }
            return;
          }
        }
        // If we get here then we can handle the response now
        Mapping::Mapper::TaskProfilingInfo info;
        info.profiling_responses.attach_realm_profiling_response(response);
        info.task_response = task_prof->task; 
        info.region_requirement_index = task_prof->src;
        info.total_reports = outstanding_profiling_requests.load();
        info.fill_response = task_prof->fill;
        if (info.task_response)
        {
          // If we had an overhead profiler
          // see if this is the callback for the task
          if (execution_context->overhead_profiler!= NULL)
            // This is the callback for the task itself
            info.profiling_responses.attach_overhead(
                execution_context->overhead_profiler);
        }
        mapper->invoke_task_report_profiling(this, info);
      }
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    } 

    //--------------------------------------------------------------------------
    void SingleTask::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void SingleTask::finalize_single_task_profiling(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(profiling_reported.exists());
#endif
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      if (outstanding_profiling_requests.load() > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapped_event.has_triggered());
#endif
        std::vector<SingleProfilingInfo> to_perform;
        {
          AutoLock o_lock(op_lock);
          to_perform.swap(profiling_info);
        }
        if (!to_perform.empty())
        {
          for (unsigned idx = 0; idx < to_perform.size(); idx++)
          {
            SingleProfilingInfo &info = to_perform[idx];
            const Realm::ProfilingResponse resp(info.buffer,info.buffer_size);
            info.total_reports = outstanding_profiling_requests.load();
            info.profiling_responses.attach_realm_profiling_response(resp);
            mapper->invoke_task_report_profiling(this, info);
            free(info.buffer);
          }
          const int count = to_perform.size() + 
              outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
          assert(count <= outstanding_profiling_requests.load());
#endif
          if (count == outstanding_profiling_requests.load())
            Runtime::trigger_event(profiling_reported);
        }
      }
      else
      {
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::TaskProfilingInfo info;
        info.total_reports = 0;
        info.task_response = true;
        info.region_requirement_index = 0;
        info.fill_response = false; // make valgrind happy
        mapper->invoke_task_report_profiling(this, info);    
        Runtime::trigger_event(profiling_reported);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::handle_remote_profiling_response(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      size_t buffer_size;
      derez.deserialize(buffer_size);
      const void *buffer = derez.get_current_pointer();
      derez.advance_pointer(buffer_size);
#ifdef DEBUG_LEGION
      // Realm needs this buffer to have 8-byte alignment so check that it does
      assert((uintptr_t(buffer) % 8) == 0);
#endif
      bool has_tracker;
      derez.deserialize(has_tracker);
      Mapping::ProfilingMeasurements::RuntimeOverhead tracker;
      if (has_tracker)
        derez.deserialize(tracker);
      const Realm::ProfilingResponse response(buffer, buffer_size);
      const OpProfilingResponse *task_prof = 
            static_cast<const OpProfilingResponse*>(response.user_data());
      Mapping::Mapper::TaskProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.task_response = task_prof->task; 
      info.region_requirement_index = task_prof->src;
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = task_prof->fill;
      if (has_tracker)
        info.profiling_responses.attach_overhead(&tracker);
      mapper->invoke_task_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SingleTask::process_remote_profiling_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      SingleTask *target;
      derez.deserialize(target);
      target->handle_remote_profiling_response(derez);
    }

    //--------------------------------------------------------------------------
    TaskContext* SingleTask::create_execution_context(VariantImpl *v,
        std::set<ApEvent> &launch_events, bool inline_task, bool leaf_task)
    //--------------------------------------------------------------------------
    {
      if (!leaf_task)
      {
        InnerContext *inner_ctx = new InnerContext(runtime, this, 
            get_depth(), v->is_inner(), regions, output_regions,
            parent_req_indexes, virtual_mapped, execution_fence_event, 0/*did*/, 
            inline_task,concurrent_task || parent_ctx->is_concurrent_context());
        configure_execution_context(inner_ctx);
        inner_ctx->add_base_gc_ref(SINGLE_TASK_REF);
        return inner_ctx;
      }
      else
      {
        LeafContext *leaf_ctx = new LeafContext(runtime, this, inline_task);
        leaf_ctx->add_base_gc_ref(SINGLE_TASK_REF);
        return leaf_ctx;
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::record_inner_termination(ApEvent termination_event)
    //--------------------------------------------------------------------------
    {
      if (single_task_termination.exists())
      {
        Runtime::trigger_event(NULL, 
            single_task_termination, termination_event); 
      }
      else
      {
        if (termination_event.exists())
        {
          AutoLock o_lock(op_lock);
          task_completion_effects.insert(termination_event);
        }
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void SingleTask::handle_deferred_task_complete(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferTriggerTaskCompleteArgs *targs =
        (const DeferTriggerTaskCompleteArgs*)args;
      targs->task->trigger_task_complete();
    }

    //--------------------------------------------------------------------------
    /*static*/ void SingleTask::order_concurrent_task_launch(const void *args)
    //--------------------------------------------------------------------------
    {
      const OrderConcurrentLaunchArgs *oargs = 
        (const OrderConcurrentLaunchArgs*)args;
      oargs->task->runtime->order_concurrent_task_launch(oargs->processor,
          oargs->task, oargs->start, oargs->ready, oargs->vid); 
    }

    /////////////////////////////////////////////////////////////
    // Multi Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MultiTask::MultiTask(Runtime *rt)
      : CollectiveViewCreator<TaskOp>(rt)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    MultiTask::~MultiTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void MultiTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, ACTIVATE_MULTI_CALL);
      CollectiveViewCreator<TaskOp>::activate();
      launch_space = NULL;
      future_map_coordinate = 0;
      future_handles = NULL;
      internal_space = IndexSpace::NO_SPACE;
      sliced = false;
      redop = 0;
      deterministic_redop = false;
      reduction_op = NULL;
      serdez_redop_fns = NULL;
      serdez_redop_state = NULL;
      serdez_redop_state_size = 0;
      reduction_metadata = NULL;
      reduction_metasize = 0;
      reduction_instance = NULL;
      first_mapping = true;
      concurrent_verified = RtUserEvent::NO_RT_USER_EVENT;
      concurrent_task_barrier = RtBarrier::NO_RT_BARRIER;
      children_complete_invoked = false;
      children_commit_invoked = false;
      predicate_false_result = NULL;
      predicate_false_size = 0;
      concurrent_lamport_clock = 0;
      concurrent_variant = 0;
      concurrent_poisoned = false;
    }

    //--------------------------------------------------------------------------
    void MultiTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, DEACTIVATE_MULTI_CALL);
      if (runtime->profiler != NULL)
        runtime->profiler->register_multi_task(this, task_id);
      CollectiveViewCreator<TaskOp>::deactivate(freeop);
      if (remove_launch_space_reference(launch_space))
        delete launch_space;
      if ((future_handles != NULL) && future_handles->remove_reference())
        delete future_handles;
      redop_initial_value = Future();
      // Remove our reference to the future map
      future_map = FutureMap();
      if (reduction_instance != NULL)
        delete reduction_instance.load();
      reduction_fold_effects.clear();
      if (serdez_redop_state != NULL)
        free(serdez_redop_state);
      if (reduction_metadata != NULL)
        free(reduction_metadata);
      if (!temporary_futures.empty())
      {
        for (std::map<DomainPoint,
              std::pair<FutureInstance*,ApEvent> >::const_iterator it =
              temporary_futures.begin(); it != temporary_futures.end(); it++)
          delete it->second.first;
        temporary_futures.clear();
      }
      concurrent_processors.clear();
      // Remove our reference to the point arguments 
      point_arguments = FutureMap();
      point_futures.clear();
      output_region_options.clear();
      output_region_extents.clear();
      slices.clear(); 
      if (predicate_false_result != NULL)
      {
        legion_free(PREDICATE_ALLOC, predicate_false_result, 
                    predicate_false_size);
        predicate_false_result = NULL;
        predicate_false_size = 0;
      }
      predicate_false_future = Future();
      intra_space_dependences.clear();
    }

    //--------------------------------------------------------------------------
    bool MultiTask::is_sliced(void) const
    //--------------------------------------------------------------------------
    {
      return sliced;
    }

    //--------------------------------------------------------------------------
    void MultiTask::slice_index_space(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_INDEX_SPACE_CALL);
#ifdef DEBUG_LEGION
      assert(!sliced);
#endif
      sliced = true;
      stealable = false; // cannot steal something that has been sliced
      Mapper::SliceTaskInput input;
      Mapper::SliceTaskOutput output;
      input.domain_is = internal_space;
      if (sharding_space.exists())
        input.sharding_is = sharding_space;
      else
        input.sharding_is = launch_space->handle;
      runtime->forest->find_domain(internal_space, input.domain);
      output.verify_correctness = false;
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_slice_task(this, input, output);
      if (output.slices.empty())
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'slice_task' "
                      "call on mapper %s. Mapper failed to specify an slices "
                      "for task %s (ID %lld).", mapper->get_mapper_name(),
                      get_task_name(), get_unique_id())
#ifdef DEBUG_LEGION
      size_t total_points = 0;
#endif
      for (unsigned idx = 0; idx < output.slices.size(); idx++)
      {
        Mapper::TaskSlice &slice = output.slices[idx]; 
        if (!slice.proc.exists())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'slice_task' "
                        "on mapper %s. Mapper returned a slice for task "
                        "%s (ID %lld) with an invalid processor " IDFMT ".",
                        mapper->get_mapper_name(), get_task_name(),
                        get_unique_id(), slice.proc.id)
        // Check to see if we need to get an index space for this domain
        if (!slice.domain_is.exists() && (slice.domain.get_volume() > 0))
          slice.domain_is = 
            runtime->find_or_create_index_slice_space(slice.domain,
                  internal_space.get_type_tag(), get_provenance());
        if (slice.domain_is.get_type_tag() != internal_space.get_type_tag())
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'slice_task' "
                        "on mapper %s. Mapper returned slice index space %d "
                        "for task %s (UID %lld) with a different type than "
                        "original index space to be sliced.",
                        mapper->get_mapper_name(), slice.domain_is.get_id(),
                        get_task_name(), get_unique_id());
#ifdef DEBUG_LEGION
        // Check to make sure the domain is not empty
        Domain &d = slice.domain;
        if ((d == Domain::NO_DOMAIN) && slice.domain_is.exists())
          runtime->forest->find_domain(slice.domain_is, d);
        bool empty = false;
	size_t volume = d.get_volume();
	if (volume == 0)
	  empty = true;
	else
	  total_points += volume;
        if (empty)
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                        "Invalid mapper output from invocation of 'slice_task' "
                        "on mapper %s. Mapper returned an empty slice for task "
                        "%s (ID %lld).", mapper->get_mapper_name(),
                        get_task_name(), get_unique_id())
#endif
        SliceTask *new_slice = this->clone_as_slice_task(slice.domain_is,
                                                         slice.proc,
                                                         slice.recurse,
                                                         slice.stealable);
        slices.push_back(new_slice);
      }
#ifdef DEBUG_LEGION
      // If the volumes don't match, then something bad happend in the mapper
      if (total_points != input.domain.get_volume())
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of 'slice_task' "
                      "on mapper %s. Mapper returned slices with a total "
                      "volume %ld that does not match the expected volume of "
                      "%zd when slicing task %s (ID %lld).", 
                      mapper->get_mapper_name(), long(total_points),
                      input.domain.get_volume(), 
                      get_task_name(), get_unique_id())
#endif
      if (output.verify_correctness)
      {
        std::vector<IndexSpace> slice_spaces(slices.size());
        for (unsigned idx = 0; idx < output.slices.size(); idx++)
          slice_spaces[idx] = output.slices[idx].domain_is;
        runtime->forest->validate_slicing(internal_space, slice_spaces,
                                          this, mapper);
      }
      trigger_slices(); 
      // If we succeeded and this is an intermediate slice task
      // then we can reclaim it, otherwise, if it is the original
      // index task then we want to keep it around. Note it is safe
      // to call get_task_kind here despite the cleanup race because
      // it is a static property of the object.
      if (get_task_kind() == SLICE_TASK_KIND)
        deactivate();
    }

    //--------------------------------------------------------------------------
    void MultiTask::trigger_slices(void)
    //--------------------------------------------------------------------------
    {
      // Add our slices back into the queue of things that are ready to map
      // or send it to its remote node if necessary
      // Watch out for the cleanup race with some acrobatics here
      // to handle the case where the iterator is invalidated
      std::set<RtEvent> wait_for;
      std::list<SliceTask*>::const_iterator it = slices.begin();
      while (true)
      {
        SliceTask *slice = *it;
        // Have to update this before launching the task to avoid 
        // the clean-up race
        it++;
        const bool done = (it == slices.end());
        // Dumb case for must epoch operations, we need these to 
        // be mapped immediately, mapper be damned
        if (must_epoch != NULL)
        {
          TriggerTaskArgs trigger_args(slice);
          RtEvent done = runtime->issue_runtime_meta_task(trigger_args, 
                                           LG_THROUGHPUT_WORK_PRIORITY);
          wait_for.insert(done);
        }
        // If we're replaying this for for a trace then don't even
        // bother asking the mapper about when to map this
        else if (is_replaying())
          slice->enqueue_ready_operation();
        // Figure out whether this task is local or remote
        else if (!runtime->is_local(slice->target_proc))
        {
          // We can only send it away if it is not origin mapped
          // otherwise it has to stay here until it is fully mapped
          if (!slice->is_origin_mapped())
            runtime->send_task(slice);
          else
            slice->enqueue_ready_task(false/*use target*/);
        }
        else
          slice->enqueue_ready_task(true/*use target*/);
        if (done)
          break;
      }
      // Must-epoch operations are nasty little beasts and have
      // to wait for the effects to finish before returning
      if (!wait_for.empty())
      {
        RtEvent wait_on = Runtime::merge_events(wait_for);
        wait_on.wait();
      }
    }

    //--------------------------------------------------------------------------
    void MultiTask::clone_multi_from(MultiTask *rhs, IndexSpace is,
                                     Processor p, bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, CLONE_MULTI_CALL);
#ifdef DEBUG_LEGION
      assert(this->launch_space == NULL);
      assert(this->future_handles == NULL);
#endif
      this->clone_task_op_from(rhs, p, stealable, false/*duplicate*/);
      this->index_domain = rhs->index_domain;
      this->launch_space = rhs->launch_space;
      add_launch_space_reference(this->launch_space);
      this->future_map_coordinate = rhs->future_map_coordinate;
      this->future_handles = rhs->future_handles;
      if (this->future_handles != NULL)
        this->future_handles->add_reference();
      this->internal_space = is;
      this->future_map = rhs->future_map;
      this->must_epoch_task = rhs->must_epoch_task;
      this->sliced = !recurse;
      this->redop = rhs->redop;
      if (this->redop != 0)
      {
        this->reduction_op = rhs->reduction_op;
        this->deterministic_redop = rhs->deterministic_redop;
        if (!this->deterministic_redop)
        {
          // Only need to initialize this if we're not doing a 
          // deterministic reduction operation
          this->serdez_redop_fns = rhs->serdez_redop_fns;
        }
      }
      this->point_arguments = rhs->point_arguments;
      if (!rhs->point_futures.empty())
        this->point_futures = rhs->point_futures;
      this->output_region_options = rhs->output_region_options;
      this->output_region_extents.resize(this->output_region_options.size());
      if (!elide_future_return)
      {
        this->predicate_false_future = rhs->predicate_false_future;
        this->predicate_false_size = rhs->predicate_false_size;
        if (this->predicate_false_size > 0)
        {
#ifdef DEBUG_LEGION
          assert(this->predicate_false_result == NULL);
#endif
          this->predicate_false_result = malloc(this->predicate_false_size);
          memcpy(this->predicate_false_result, rhs->predicate_false_result,
                 this->predicate_false_size);
        }
      }
    }

    //--------------------------------------------------------------------------
    Domain MultiTask::get_slice_domain(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(internal_space.exists());
#endif
      Domain result;
      runtime->forest->find_domain(internal_space, result);
      return result; 
    }

    //--------------------------------------------------------------------------
    void MultiTask::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, MULTI_TRIGGER_EXECUTION_CALL);
      if (is_remote())
      {
        // distribute, slice, then map/launch
        if (distribute_task())
        {
          // Still local
          if (is_sliced())
          {
            if (is_origin_mapped())
              launch_task();
            else
              map_and_launch();
          }
          else
            slice_index_space();
        }
      }
      else
      {
        // Not remote
        if (must_epoch == NULL)
          premap_task();
        if (is_origin_mapped())
        {
          if (is_sliced())
          {
            if (must_epoch == NULL)
            {
              // See if we've done our first mapping yet or not
              if (first_mapping)
              {
                first_mapping = false;
                const RtEvent done_mapping = perform_mapping();
                if (!done_mapping.exists() || done_mapping.has_triggered())
                {
                  if (distribute_task())
                    launch_task();
                }
                else
                  defer_distribute_task(done_mapping);
              }
              else
              {
                // We know that it is staying on one
                // of our local processors.  If it is
                // still this processor then map and run it
                if (distribute_task())
                  // Still local so we can launch it
                  launch_task();
              }
            }
            else
              register_must_epoch();
          }
          else
            slice_index_space();
        }
        else
        {
          if (distribute_task())
          {
            // Still local try slicing, mapping, and launching
            if (is_sliced())
              map_and_launch();
            else
              slice_index_space();
          }
        }
      }
    } 

    //--------------------------------------------------------------------------
    void MultiTask::pack_multi_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, PACK_MULTI_CALL);
      RezCheck z(rez);
      pack_base_task(rez, target);
      rez.serialize(launch_space->handle);
      rez.serialize(sliced);
      rez.serialize(redop);
      if (redop > 0)
        rez.serialize<bool>(deterministic_redop);
      else if (future_handles != NULL)
      {
        // Only pack the IDs for our local points
        IndexSpaceNode *node = runtime->forest->get_node(internal_space);
        Domain local_domain;
        node->get_domain(local_domain);
        size_t local_size = local_domain.get_volume();
        rez.serialize(local_size);
        const std::map<DomainPoint,DistributedID> &handles =
          future_handles->handles;
#ifdef DEBUG_LEGION
        assert(local_size <= handles.size());
#endif
        if (local_size < handles.size())
        {
          for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
          {
            std::map<DomainPoint,DistributedID>::const_iterator finder = 
              handles.find(itr.p);
#ifdef DEBUG_LEGION
            assert(finder != handles.end());
#endif
            rez.serialize(finder->first);
            rez.serialize(finder->second);
          }
        }
        else
        {
          for (std::map<DomainPoint,DistributedID>::const_iterator it =
                handles.begin(); it != handles.end(); it++)
          {
            rez.serialize(it->first);
            rez.serialize(it->second);
          }
        }
        rez.serialize(future_map_coordinate);
      }
      else
        rez.serialize<size_t>(0);
      if (!output_region_options.empty())
      {
        rez.serialize<size_t>(output_region_options.size());
        for (unsigned idx = 0; idx < output_region_options.size(); idx++)
          rez.serialize(output_region_options[idx]);
      }
      else
        rez.serialize<size_t>(0); 
    }

    //--------------------------------------------------------------------------
    void MultiTask::unpack_multi_task(Deserializer &derez,
                                      std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, UNPACK_MULTI_CALL);
      DerezCheck z(derez);
      unpack_base_task(derez, ready_events); 
      IndexSpace launch_handle;
      derez.deserialize(launch_handle);
#ifdef DEBUG_LEGION
      assert(launch_space == NULL);
#endif
      launch_space = runtime->forest->get_node(launch_handle);
      add_launch_space_reference(launch_space);
      derez.deserialize(sliced);
      derez.deserialize(redop);
      if (redop > 0)
      {
        reduction_op = Runtime::get_reduction_op(redop);
        derez.deserialize(deterministic_redop);
        if (!deterministic_redop)
        {
          // Only need to fill this in if we're not doing a 
          // deterministic reduction operation
          serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(future_handles == NULL);
#endif
        size_t num_handles;
        derez.deserialize(num_handles);
        if (num_handles > 0)
        {
          future_handles = new FutureHandles;
          future_handles->add_reference();
          std::map<DomainPoint,DistributedID> &handles = 
            future_handles->handles;
          for (unsigned idx = 0; idx < num_handles; idx++)
          {
            DomainPoint point;
            derez.deserialize(point);
            derez.deserialize(handles[point]);
          }
          derez.deserialize(future_map_coordinate);
        }
      }
      size_t num_globals;
      derez.deserialize(num_globals);
      if (num_globals > 0)
      {
        output_region_options.resize(num_globals);
        for (unsigned idx = 0; idx < num_globals; idx++)
          derez.deserialize(output_region_options[idx]);
        output_region_extents.resize(num_globals);
      }
    }

    //--------------------------------------------------------------------------
    bool MultiTask::fold_reduction_future(FutureInstance *instance, 
                                          ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // Apply the reduction operation
#ifdef DEBUG_LEGION
      assert(reduction_op != NULL);
#endif
      // Perform the reduction, see if we have to do serdez reductions
      if (serdez_redop_fns != NULL)
      {
        // If this instance is not meta-visible we need to copy
        // it to a local buffer here
        FutureInstance *bounce_instance = NULL;
        if (!instance->is_meta_visible)
        {
#ifdef __GNUC__
#if __GNUC__ >= 11
          // GCC is dumb and thinks we need to initialize this buffer
          // before we pass it into the create local call, which we
          // obviously don't need to do, so tell the compiler to shut up
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif
          void *bounce_buffer = malloc(instance->size);
          bounce_instance = FutureInstance::create_local(bounce_buffer,
                                          instance->size, true/*own*/);
#ifdef __GNUC__
#if __GNUC__ >= 11
#pragma GCC diagnostic pop
#endif
#endif
          // Wait for the data here to be ready
          const ApEvent ready = bounce_instance->copy_from(instance, this, effects);
          if (ready.exists())
          {
            bool poisoned = false;
            ready.wait_faultaware(poisoned);
            if (poisoned)
              parent_ctx->raise_poison_exception();
          }
          instance = bounce_instance;
        }
        // Need to lock to make the serialize/deserialize process atomic
        {
          AutoLock o_lock(op_lock);
          // See if we're the first one to get here
          if (serdez_redop_state == NULL)
            (*(serdez_redop_fns->init_fn))(reduction_op, serdez_redop_state,
                                           serdez_redop_state_size);
          (*(serdez_redop_fns->fold_fn))(reduction_op, serdez_redop_state,
                             serdez_redop_state_size, instance->get_data());
        }
        if (bounce_instance != NULL)
          delete bounce_instance;
        return true;
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(reduction_instance != NULL); 
#endif
        if (effects.exists())
        {
          if (reduction_instance_precondition.exists())
            effects = Runtime::merge_events(NULL, effects, 
                reduction_instance_precondition);
        }
        else
          effects = reduction_instance_precondition;
        if (!deterministic_redop)
        {
          const ApEvent done = reduction_instance.load()->reduce_from(instance,
              this, redop, reduction_op, false/*exclusive*/, effects);
          if (done.exists())
          {
            AutoLock o_lock(op_lock);
            reduction_fold_effects.push_back(done);
            return false;
          }
          else
            return true;
        }
        else
        {
          // No need for the lock since we know the caller is ensuring order
          reduction_instance_precondition = 
            reduction_instance.load()->reduce_from(instance, this, redop,
              reduction_op, true/*exclusive*/, effects);
          return !reduction_instance_precondition.exists();
        }
      }
    } 

    /////////////////////////////////////////////////////////////
    // Individual Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndividualTask::IndividualTask(Runtime *rt)
      : SingleTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndividualTask::IndividualTask(const IndividualTask &rhs)
      : SingleTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndividualTask::~IndividualTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndividualTask& IndividualTask::operator=(const IndividualTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, ACTIVATE_INDIVIDUAL_CALL);
      SingleTask::activate();
      output_regions_registered = RtEvent::NO_RT_EVENT;
      predicate_false_result = NULL;
      predicate_false_size = 0;
      orig_task = this;
      remote_unique_id = get_unique_id();
      sent_remotely = false;
      top_level_task = false;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, DEACTIVATE_INDIVIDUAL_CALL);
      SingleTask::deactivate(false/*free*/);
      if (predicate_false_result != NULL)
      {
        legion_free(PREDICATE_ALLOC, predicate_false_result, 
                    predicate_false_size);
        predicate_false_result = NULL;
        predicate_false_size = 0;
      }
      // Remove our reference on the future
      result = Future();
      predicate_false_future = Future();
      valid_output_regions.clear();
      if (freeop)
        runtime->free_individual_task(this);
    }

    //--------------------------------------------------------------------------
    Future IndividualTask::initialize_task(InnerContext *ctx,
                                           const TaskLauncher &launcher,
                                           Provenance *provenance,
                                           bool top_level /*=false*/,
                                           bool must_epoch_launch /*=false*/,
                              std::vector<OutputRequirement> *outputs /*=NULL*/)
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
        args = legion_malloc(TASK_ARGS_ALLOC, arglen);
        memcpy(args,launcher.argument.get_ptr(),arglen);
      }
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
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
      if (outputs != NULL)
      {
        create_output_regions(*outputs);
        if (launcher.predicate != Predicate::TRUE_PRED)
          REPORT_LEGION_ERROR(ERROR_OUTPUT_REGIONS_IN_PREDICATED_TASK,
              "Output requirements are disallowed for tasks launched with "
              "predicates, but preidcated task launch for task %s (%lld) in "
              "parent task %s (UID %lld) is used with output requirements.",
              get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
        if (get_trace() != NULL)
          REPORT_LEGION_ERROR(ERROR_OUTPUT_REGIONS_IN_TRACE,
              "Output requirements are disallowed for tasks launched inside "
              "traces. Task %s (UID %lld) in parent task %s (UID %lld) has "
              "output requirements in trace %d.", get_task_name(), 
              get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id(), get_trace()->get_trace_id())
      }
      if (launcher.predicate != Predicate::TRUE_PRED &&
          !launcher.elide_future_return)
      {
        if (launcher.predicate_false_future.impl != NULL)
          predicate_false_future = launcher.predicate_false_future;
        else
        {
          predicate_false_size = launcher.predicate_false_result.get_size();
          if (predicate_false_size > 0)
          {
#ifdef DEBUG_LEGION
            assert(predicate_false_result == NULL);
#endif
            predicate_false_result = 
              legion_malloc(PREDICATE_ALLOC, predicate_false_size);
            memcpy(predicate_false_result, 
                   launcher.predicate_false_result.get_ptr(),
                   predicate_false_size);
          }
        }
      } 
      if (launcher.local_function_task)
      {
        if (!regions.empty())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_LOCAL_FUNCTION_TASK_LAUNCH, 
              "Local function task launch for task %s in parent task %s "
              "(UID %lld) has %zd region requirements. Local function tasks "
              "are not permitted to have any region requirements.", 
              get_task_name(), parent_ctx->get_task_name(), 
              parent_ctx->get_unique_id(), regions.size())
        local_function = true;
      }
      // Get a future from the parent context to use as the result
      if (launcher.elide_future_return)
        elide_future_return = true;
      else if (!must_epoch_launch)
        result = create_future();
      validate_region_requirements(); 
      // If this is the top-level task we can record some extra properties
      if (top_level)
        this->top_level_task = true;
      if (runtime->legion_spy_enabled)
      {
        if (top_level)
          LegionSpy::log_top_level_task(task_id, parent_ctx->get_unique_id(),
                                        unique_op_id, get_task_name());
        // Tracking as long as we are not part of a must epoch operation
        if (!must_epoch_launch || top_level)
          LegionSpy::log_individual_task(parent_ctx->get_unique_id(),
                                         unique_op_id, task_id,
                                         get_task_name());
        for (std::vector<PhaseBarrier>::const_iterator it = 
              launcher.wait_barriers.begin(); it !=
              launcher.wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
      }
#ifdef DEBUG_LEGION
      if (!launcher.independent_requirements)
        perform_intra_task_alias_analysis();
#endif
      return result;
    }

    //--------------------------------------------------------------------------
    Future IndividualTask::create_future(void)
    //--------------------------------------------------------------------------
    {
      FutureImpl *impl = new FutureImpl(parent_ctx, runtime, true/*register*/,
              runtime->get_available_distributed_id(), get_provenance(), this);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_future_creation(unique_op_id, 
                                       impl->did, index_point);
      return Future(impl);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::prepare_map_must_epoch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch != NULL);
#endif
      set_origin_mapped(true);
      if (!elide_future_return)
      {
        FutureMap map = must_epoch->get_future_map();
        result = map.impl->get_future(index_point, true/*internal only*/);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::create_output_regions(
                                        std::vector<OutputRequirement> &outputs)
    //--------------------------------------------------------------------------
    {
      valid_output_regions.resize(outputs.size());
      Provenance *provenance = get_provenance();
      for (unsigned idx = 0; idx < outputs.size(); idx++)
      {
        OutputRequirement &req = outputs[idx];
        valid_output_regions[idx] = req.valid_requirement;

        if (!req.valid_requirement)
        {
          // Create a deferred index space
          IndexSpace index_space =
            parent_ctx->create_unbound_index_space(req.type_tag, provenance);
          // Create an output region
          LogicalRegion region = parent_ctx->create_logical_region(
              index_space, req.field_space, false/*local region*/,
              provenance, true/*output region*/);

          // Set the region back to the output requirement so the caller
          // can use it for downstream tasks
          req.region = region;
          req.parent = region;
          req.flags |= LEGION_CREATED_OUTPUT_REQUIREMENT_FLAG;
        }
        req.privilege = LEGION_WRITE_DISCARD;

        // Store the output requirement in the task
        output_regions.push_back(req);
      }
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      if (!options_selected)
      {
        const bool inline_task = select_task_options(false/*prioritize*/);
        if (inline_task) 
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_INLINE,
                          "Mapper %s requested to inline task %s "
                          "(UID %lld) but the 'enable_inlining' option was "
                          "not set on the task launcher so the request is "
                          "being ignored", mapper->get_mapper_name(),
                          get_task_name(), get_unique_id());
        }
      }
      // local function tasks have no region requirements so nothing below
      if (local_function)
        return;
      // First compute the parent indexes
      compute_parent_indexes();
      update_no_access_regions();
      if (runtime->legion_spy_enabled)
      {
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
          log_requirement(unique_op_id, idx, logical_regions[idx]);
      }
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
#ifdef DEBUG_LEGION
      assert(memo_state != MEMO_REQ);
#endif
      if (runtime->check_privileges && 
          !is_top_level_task() && !local_function)
        perform_privilege_checks();
      // To be correct with the new scheduler we also have to 
      // register mapping dependences on futures
      for (std::vector<Future>::const_iterator it = futures.begin();
            it != futures.end(); it++)
        if (it->impl != NULL)
          it->impl->register_dependence(this);
      if (predicate_false_future.impl != NULL)
        predicate_false_future.impl->register_dependence(this);
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                  wait_barriers, arrive_barriers, must_epoch);
      version_infos.resize(logical_regions.size());
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Dumb case for must epoch operations, we need these to 
      // be mapped immediately, mapper be damned
      if (must_epoch != NULL)
      {
        TriggerTaskArgs trigger_args(this);
        runtime->issue_runtime_meta_task(trigger_args, 
                                         LG_THROUGHPUT_WORK_PRIORITY);
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
          enqueue_ready_task(false/*use target*/);
      }
      else
        enqueue_ready_task(true/*use target*/);
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::report_interfering_requirements(unsigned idx1, 
                                                         unsigned idx2)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ALIASED_INTERFERING_REGION,
                    "Aliased and interfering region requirements for "
                    "individual tasks are not permitted. Region requirements "
                    "%d and %d of task %s (UID %lld) in parent task %s "
                    "(UID %lld) are interfering.", idx1, idx2, get_task_name(),
                    get_unique_id(), parent_ctx->get_task_name(),
                    parent_ctx->get_unique_id())
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      complete_mapping();
      if (!elide_future_return)
      {
        // Set the future to the false result
        if (predicate_false_future.impl == NULL)
        {
          if (predicate_false_size > 0)
            result.impl->set_local(predicate_false_result,
                                   predicate_false_size, false/*own*/);
          else
            result.impl->set_result(ApEvent::NO_AP_EVENT, NULL);
        }
        else
          result.impl->set_result(predicate_false_future.impl, this);
      }
      complete_execution();
      trigger_children_complete();
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      if (target_proc.exists() && (target_proc != current_proc))
      {
        runtime->send_task(this);
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent IndividualTask::perform_mapping(
                                        MustEpochOp *must_epoch_owner/*=NULL*/, 
                                        const DeferMappingArgs *args/* =NULL*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_PERFORM_MAPPING_CALL);
      const RtEvent deferred = map_all_regions(must_epoch_owner, args);
      if (deferred.exists())
        return deferred; 
      // If we mapped, then we are no longer stealable
      stealable = false;
      // If we're remote, send back a message to the origin node instance
      // of the task to tell it that we are mapped
      if (is_remote())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize<SingleTask*>(orig_task);
          rez.serialize(mapped_event);
        }
        runtime->send_individual_remote_mapped(orig_proc, rez);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::replicate_task(void)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        // Pull these onto the stack since it's unsafe to read them after
        // we call replicate task and it goes off and does stuff
        SingleTask *original = orig_task;
        const Processor orig = orig_proc;
        const RtEvent event = mapped_event;
        const bool result = SingleTask::replicate_task();
        if (result)
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize<SingleTask*>(original);
            rez.serialize(event);
          }
          runtime->send_individual_remote_mapped(orig, rez);
        }
        return result;
      }
      else
        return SingleTask::replicate_task();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_future_size(size_t return_type_size,
                   bool has_return_type_size, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if (elide_future_return)
        return;
      if (is_remote())
      {
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(orig_task);
          rez.serialize(return_type_size);
          rez.serialize(has_return_type_size);
          rez.serialize(done_event);
        }
        runtime->send_individual_remote_future_size(orig_proc, rez);
        applied_events.insert(done_event);
      }
      else if (has_return_type_size)
        result.impl->set_future_result_size(return_type_size,
                                            runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::record_output_registered(RtEvent registered,
                                              std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(registered.exists());
#endif
      if (is_remote())
      {
        // Send the message on to the origin node to tell it
        // to launch the meta task to perform the registration
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(orig_task);
          rez.serialize(registered);
          rez.serialize(applied);
        }
        runtime->send_individual_remote_output_registration(orig_proc, rez);
        applied_events.insert(applied);
      }
      else
      {
        // Launch the meta-task to perform the registration
        // Make sure we don't complete the task until this is done
        FinalizeOutputEqKDTreeArgs args(this);
        output_regions_registered = 
          runtime->issue_runtime_meta_task(args,
              LG_LATENCY_DEFERRED_PRIORITY, registered);
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::handle_remote_output_registration(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndividualTask *task;
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
    void IndividualTask::perform_inlining(VariantImpl *variant,
                                  const std::deque<InstanceSet> &parent_regions)
    //--------------------------------------------------------------------------
    {
      SingleTask::perform_inlining(variant, parent_regions);
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
      return valid_output_regions[idx];
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind IndividualTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INDIVIDUAL_TASK_KIND;
    } 

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_TRIGGER_COMPLETE_CALL);
      // Invalidate any state that we had if we didn't already
      // Do this before sending the complete message to avoid the
      // race condition in the remote case where the top-level
      // context cleans on the owner node while we still need it
      std::set<RtEvent> completion_preconditions;
      if (execution_context != NULL)
      {
        execution_context->invalidate_region_tree_contexts(is_top_level_task(),
                                                      completion_preconditions);
        if (runtime->legion_spy_enabled)
          execution_context->log_created_requirements();
      }
      // For remote cases we have to keep track of the events for
      // returning any created logical state, we can't commit until
      // it is returned or we might prematurely release the references
      // that we hold on the version state objects
      if (!is_remote())
      {
        // Pass back our created and deleted operations
        if (execution_context != NULL)
        {
          if (top_level_task)
            execution_context->report_leaks_and_duplicates(
                                  completion_preconditions);
          else if (must_epoch != NULL)
            execution_context->return_resources(must_epoch,
                   context_index, completion_preconditions);
          else
            execution_context->return_resources(parent_ctx, 
                   context_index, completion_preconditions);
        }
        if (output_regions_registered.exists())
          completion_preconditions.insert(output_regions_registered);
      }
      else
      {
        Serializer rez;
        if (!completion_preconditions.empty())
        {
          const RtEvent complete_precondition =
            Runtime::merge_events(completion_preconditions);
          pack_remote_complete(rez, complete_precondition);
        }
        else
          pack_remote_complete(rez, RtEvent::NO_RT_EVENT);
        runtime->send_individual_remote_complete(orig_proc,rez);
      }
      // See if we need to trigger that our children are complete
      // Note it is only safe to do this if we were not sent remotely
      bool need_commit = false;
      if (!sent_remotely && (execution_context != NULL))
        need_commit = execution_context->attempt_children_commit();
      if (must_epoch != NULL)
      {
        RtEvent precondition;
        if (!completion_preconditions.empty())
          precondition = Runtime::merge_events(completion_preconditions);
        if (!task_completion_effects.empty())
          must_epoch->record_completion_effects(task_completion_effects);
        must_epoch->notify_subop_complete(this, precondition);
        complete_operation();
      }
      else
      {
        if (!task_completion_effects.empty())
          Operation::record_completion_effects(task_completion_effects);
        if (!completion_preconditions.empty())
          complete_operation(Runtime::merge_events(completion_preconditions));
        else
          complete_operation();
      }
      if (need_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_TRIGGER_COMMIT_CALL);
      if (is_remote())
      {
        Serializer rez;
        pack_remote_commit(rez);
        runtime->send_individual_remote_commit(orig_proc,rez);
      }
      if (profiling_reported.exists())
        finalize_single_task_profiling();
      if (must_epoch != NULL)
      {
        must_epoch->notify_subop_commit(this, profiling_reported);
        commit_operation(true/*deactivate*/, profiling_reported);
      }
      else
        commit_operation(true/*deactivate*/, profiling_reported);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_post_execution(FutureInstance *instance,
                                       void *metadata, size_t metasize,
                                       FutureFunctor *functor,
                                       Processor future_proc, bool own_functor)
    //--------------------------------------------------------------------------
    {
      record_completion_effect(single_task_termination);
      if (functor != NULL)
      {
#ifdef DEBUG_LEGION
        assert(instance == NULL);
        assert(metadata == NULL);
#endif
        if (elide_future_return)
        {
          functor->callback_release_future();
          if (own_functor)
            delete functor;
        }
        else
          result.impl->set_result(single_task_termination, functor,
                                  own_functor, future_proc);
      }
      else
      {
        if ((instance != NULL) && (instance->size > 0) && 
            (shard_manager == NULL))
          check_future_return_bounds(instance);
        if (elide_future_return)
        {
          if ((instance != NULL) && 
              !instance->defer_deletion(single_task_termination))
            delete instance;
          if (metadata != NULL)
            free(metadata);
        }
        else
        {
          if ((instance != NULL) && (instance->size > 0) && 
              (shard_manager == NULL))
            check_future_return_bounds(instance);
          result.impl->set_result(single_task_termination, instance,
                                  metadata, metasize);
        }
      }
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::handle_mispredication(void)
    //--------------------------------------------------------------------------
    {
      // First thing: increment the meta-task counts since we decremented
      // them in case we didn't end up running
#ifdef DEBUG_LEGION
      runtime->increment_total_outstanding_tasks(
          MispredicationTaskArgs::TASK_ID, true/*meta*/);
#else
      runtime->increment_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
      runtime->outstanding_counts[MispredicationTaskArgs::TASK_ID].fetch_add(1);
#endif
      if (!elide_future_return)
      {
        // Set the future to the false result
        if (predicate_false_future.impl == NULL)
        {
          if (predicate_false_size > 0)
            result.impl->set_local(predicate_false_result,
                                   predicate_false_size, false/*own*/);
          else
            result.impl->set_result(ApEvent::NO_AP_EVENT, NULL);
        }
        else
          result.impl->set_result(predicate_false_future.impl, this);
      }
      // Pretend like we executed the task
      execution_context->handle_mispredication();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::concurrent_allreduce(ProcessorManager *manager, 
                           uint64_t lamport_clock, VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      manager->finalize_concurrent_task_order(this, lamport_clock, poisoned);  
    }

    //--------------------------------------------------------------------------
    void IndividualTask::perform_concurrent_task_barrier(void)
    //--------------------------------------------------------------------------
    {
      // No-op
    }

    //--------------------------------------------------------------------------
    bool IndividualTask::pack_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_PACK_TASK_CALL);
      // Check to see if we are stealable, if not and we have not
      // yet been sent remotely, then send the state now
      RezCheck z(rez);
      pack_single_task(rez, target);
      size_t valid_output_regions_size = valid_output_regions.size();
      rez.serialize(valid_output_regions_size);
      for (unsigned idx = 0; idx < valid_output_regions.size(); idx++)
        rez.serialize<bool>(valid_output_regions[idx]);
      rez.serialize(orig_task);
      rez.serialize(remote_unique_id);
      parent_ctx->pack_inner_context(rez);
      rez.serialize(top_level_task);
      if (!elide_future_return)
      {
        result.impl->pack_future(rez, target);
        if (predicate_false_future.impl != NULL)
          predicate_false_future.impl->pack_future(rez, target);
        else
          rez.serialize<DistributedID>(0);
        rez.serialize(predicate_false_size);
        if (predicate_false_size > 0)
          rez.serialize(predicate_false_result, predicate_false_size);
      }
      Provenance *provenance = get_provenance();
      if (provenance != NULL)
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
    bool IndividualTask::unpack_task(Deserializer &derez, Processor current,
                                     std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_UNPACK_TASK_CALL);
      DerezCheck z(derez);
      unpack_single_task(derez, ready_events);
      size_t valid_output_regions_size = 0;
      derez.deserialize(valid_output_regions_size);
      valid_output_regions.resize(valid_output_regions_size);
      for (unsigned idx = 0; idx < valid_output_regions_size; idx++)
      {
        bool valid_output_region = false;
        derez.deserialize<bool>(valid_output_region);
        valid_output_regions[idx] = valid_output_region;
      }
      derez.deserialize(orig_task);
      derez.deserialize(remote_unique_id);
      set_current_proc(current);
      // Figure out what our parent context is
      parent_ctx = InnerContext::unpack_inner_context(derez, runtime);
      derez.deserialize(top_level_task);
      // Quick check to see if we've been sent back to our original node
      if (!is_remote())
      {
#ifdef DEBUG_LEGION
        // Need to make the deserializer happy in debug mode
        // 2 *sizeof(size_t) since we're two DerezChecks deep
        derez.advance_pointer(derez.get_remaining_bytes() - 2*sizeof(size_t));
#endif
        // If we were sent back then mark that we are no longer remote
        orig_task->sent_remotely = false;
        // Put the original instance back on the mapping queue and
        // deactivate this version of the task
        orig_task->enqueue_ready_task(false/*target*/);
        deactivate();
        return false;
      }
      if (!elide_future_return)
      {
        result = FutureImpl::unpack_future(runtime, derez);
        // Unpack the predicate false infos
        predicate_false_future = FutureImpl::unpack_future(runtime, derez);
        derez.deserialize(predicate_false_size);
        if (predicate_false_size > 0)
        {
#ifdef DEBUG_LEGION
          assert(predicate_false_result == NULL);
#endif
          predicate_false_result = malloc(predicate_false_size);
          derez.deserialize(predicate_false_result, predicate_false_size);
        }
      }
      if (is_origin_mapped())
      {
        if (!is_leaf())
        {
          // Send back the event that will be triggered when the task is mapped
          Serializer rez;
          {
            RezCheck z2(rez);
            rez.serialize<SingleTask*>(orig_task);
            rez.serialize(mapped_event);
          }
          runtime->send_individual_remote_mapped(orig_proc, rez);
        }
        else 
          // We're not going to get a callback from the context if we're a leaf
          complete_mapping();
      }
      else
        version_infos.resize(logical_regions.size());
      set_provenance(Provenance::deserialize(derez));
      // Set our parent task for the user
      parent_task = parent_ctx->get_task();
      // Remote individual tasks are always resolved
      resolved = true;
      // Have to do this before resolving speculation in case
      // we get cleaned up after the resolve speculation call
      if (runtime->legion_spy_enabled)
        LegionSpy::log_point_point(remote_unique_id, get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_operation(this);
      // Return true to add ourselves to the ready queue
      return true;
    }

    //--------------------------------------------------------------------------
    void IndividualTask::record_completion_effect(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      if (!effect.exists())
        return;
      AutoLock o_lock(op_lock);
      task_completion_effects.insert(effect);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::record_completion_effect(ApEvent effect,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      if (!effect.exists())
        return;
      AutoLock o_lock(op_lock);
      task_completion_effects.insert(effect);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::record_completion_effects(
                                               const std::set<ApEvent> &effects)
    //--------------------------------------------------------------------------
    {
      if (effects.empty())
        return;
      AutoLock o_lock(op_lock);
      for (std::set<ApEvent>::const_iterator it =
            effects.begin(); it != effects.end(); it++)
        if (it->exists())
          task_completion_effects.insert(*it);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::record_completion_effects(
                                            const std::vector<ApEvent> &effects)
    //--------------------------------------------------------------------------
    {
      if (effects.empty())
        return;
      AutoLock o_lock(op_lock);
      for (std::vector<ApEvent>::const_iterator it =
            effects.begin(); it != effects.end(); it++)
        if (it->exists())
          task_completion_effects.insert(*it);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_complete(Serializer &rez, RtEvent pre)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_PACK_REMOTE_COMPLETE_CALL);
      // Send back the pointer to the task instance, then serialize
      // everything else that needs to be sent back
      rez.serialize(orig_task);
      RezCheck z(rez);
      rez.serialize(pre);
      // Pack the privilege state
      if (execution_context != NULL)
      {
        rez.serialize<bool>(true);
        execution_context->pack_return_resources(rez, context_index);
      }
      else
        rez.serialize<bool>(false);
      if (!is_origin_mapped())
      {
#ifdef DEBUG_LEGION
        // Should always include the single task termination
        assert(!task_completion_effects.empty());
#endif
        rez.serialize<size_t>(task_completion_effects.size());
        for (std::set<ApEvent>::const_iterator it =
              task_completion_effects.begin(); it != 
              task_completion_effects.end(); it++)
          rez.serialize(*it);
        // Clear it so we don't try to record it in trigger_task_complete
        task_completion_effects.clear();
      }
    }
    
    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDIVIDUAL_UNPACK_REMOTE_COMPLETE_CALL);
      DerezCheck z(derez);
      RtEvent remote_precondition;
      derez.deserialize(remote_precondition);
      // First unpack the privilege state
      bool has_privilege_state;
      derez.deserialize(has_privilege_state);
      if (has_privilege_state)
      {
        const RtEvent resources_returned = (must_epoch == NULL) ? 
          ResourceTracker::unpack_resources_return(derez, parent_ctx) :
          ResourceTracker::unpack_resources_return(derez, must_epoch);
        if (resources_returned.exists())
        {
          if (remote_precondition.exists())
            remote_precondition = 
              Runtime::merge_events(remote_precondition, resources_returned);
          else
            remote_precondition = resources_returned;
        }
      }
      if (!is_origin_mapped())
      {
        size_t num_effects;
        derez.deserialize(num_effects);
        if (num_effects > 1)
        {
          std::set<ApEvent> effects;
          for (unsigned idx = 0; idx < num_effects; idx++)
          {
            ApEvent effect;
            derez.deserialize(effect);
            effects.insert(effect);
          }
          record_completion_effects(effects);
        }
        else
        {
          ApEvent task_effects;
          derez.deserialize(task_effects);
          record_completion_effect(task_effects);
        }
      }
      // Mark that we have both finished executing and that our
      // children are complete
      complete_execution(remote_precondition);
      TaskOp::trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::pack_remote_commit(Serializer &rez)
    //--------------------------------------------------------------------------
    {
      // Only need to send back the pointer to the task instance
      rez.serialize(orig_task);
    }

    //--------------------------------------------------------------------------
    void IndividualTask::unpack_remote_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndividualTask::complete_replay(ApEvent instance_ready_event,
                                         ApEvent completion_postcondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!target_processors.empty());
#endif
      if (completion_postcondition.exists())
        record_completion_effect(completion_postcondition);
      const AddressSpaceID target_space = 
        runtime->find_address_space(target_processors.front());
      // Check to see if we're replaying this locally or remotely
      if (target_space != runtime->address_space)
      {
        // This is the remote case, pack it up and ship it over
        // Mark that we are effecitvely mapping this at the origin
        map_origin = true;
        // Pack this task up and send it to the remote node
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(instance_ready_event);
          rez.serialize(completion_postcondition);
          rez.serialize(target_processors.front());
          rez.serialize(INDIVIDUAL_TASK_KIND);
          pack_task(rez, target_space);
        }
        runtime->send_remote_task_replay(target_space, rez);
      }
      else
      { 
#ifdef DEBUG_LEGION
        assert(is_leaf());
        assert(region_preconditions.empty());
#endif
        region_preconditions.resize(regions.size(), instance_ready_event);
        execution_fence_event = instance_ready_event;
        update_no_access_regions();
        launch_task();
      }
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::process_unpack_remote_future_size(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndividualTask *task;
      derez.deserialize(task);
      size_t return_type_size;
      derez.deserialize(return_type_size);
      bool has_return_type_size;
      derez.deserialize(has_return_type_size);
      RtUserEvent done_event;
      derez.deserialize(done_event);
      std::set<RtEvent> applied_events;
      task->handle_future_size(return_type_size, 
          has_return_type_size, applied_events);
      if (!applied_events.empty())
        Runtime::trigger_event(done_event,
            Runtime::merge_events(applied_events));
      else
        Runtime::trigger_event(done_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::process_unpack_remote_mapped(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      // Single task because we share this with the point task for when
      // point tasks are origin-mapped non-leaf tasks
      SingleTask *task;
      derez.deserialize(task);
      RtEvent mapped_event;
      derez.deserialize(mapped_event);
      task->handle_post_mapped(mapped_event);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::process_unpack_remote_complete(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask *task;
      derez.deserialize(task);
      task->unpack_remote_complete(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndividualTask::process_unpack_remote_commit(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndividualTask *task;
      derez.deserialize(task);
      task->unpack_remote_commit(derez);
    }

    /////////////////////////////////////////////////////////////
    // Point Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointTask::PointTask(Runtime *rt)
      : SingleTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointTask::PointTask(const PointTask &rhs)
      : SingleTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    PointTask::~PointTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    PointTask& PointTask::operator=(const PointTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void PointTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_ACTIVATE_CALL);
      SingleTask::activate();
      orig_task = this;
      slice_owner = NULL;
      concurrent_task_barrier = RtBarrier::NO_RT_BARRIER;
    }

    //--------------------------------------------------------------------------
    void PointTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_DEACTIVATE_CALL);
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(
            this->slice_owner->get_unique_op_id(),
            this->get_unique_op_id());
      SingleTask::deactivate(false/*free*/);
      if (freeop)
        runtime->free_point_task(this);
    } 

    //--------------------------------------------------------------------------
    Operation* PointTask::get_origin_operation(void)
    //--------------------------------------------------------------------------
    {
      return slice_owner->get_origin_operation();
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_reducing_future(void) const
    //--------------------------------------------------------------------------
    {
      return slice_owner->is_reducing_future();
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      slice_owner->record_point_mapped(mapped_event);
      SingleTask::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void PointTask::report_interfering_requirements(unsigned idx1,
                                                    unsigned idx2)
    //--------------------------------------------------------------------------
    {
      switch (index_point.get_dim())
      {
        case 1:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point %lld of index space task %s (UID %lld) "
                    "in parent task %s (UID %lld) are interfering.", 
                    idx1, idx2, index_point[0], get_task_name(),
                    get_unique_id(), parent_ctx->get_task_name(),
                    parent_ctx->get_unique_id());
            break;
          }
#if LEGION_MAX_DIM >= 2
        case 2:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld) of index space task %s "
                    "(UID %lld) in parent task %s (UID %lld) are interfering.",
                    idx1, idx2, index_point[0], index_point[1], 
                    get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 3
        case 3:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld) of index space task %s"
                    " (UID %lld) in parent task %s (UID %lld) are interfering.",
                    idx1, idx2, index_point[0], index_point[1], 
                    index_point[2], get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 4
        case 4:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld) of index space " 
                    "task %s (UID %lld) in parent task %s (UID %lld) are "
                    "interfering.", idx1, idx2, index_point[0], index_point[1],
                    index_point[2], index_point[3], get_task_name(), 
                    get_unique_id(), parent_ctx->get_task_name(), 
                    parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 5
        case 5:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld) of index "
                    "space task %s (UID %lld) in parent task %s (UID %lld) are "
                    "interfering.", idx1, idx2, index_point[0], index_point[1],
                    index_point[2], index_point[3], index_point[4], 
                    get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 6
        case 6:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld,%lld) of "
                    "index space task %s (UID %lld) in parent task %s (UID "
                    "%lld) are interfering.", idx1, idx2, index_point[0], 
                    index_point[1], index_point[2], index_point[3], 
                    index_point[4], index_point[5], get_task_name(), 
                    get_unique_id(), parent_ctx->get_task_name(), 
                    parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 7
        case 7:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld,%lld,%lld) of"
                    " index space task %s (UID %lld) in parent task %s (UID "
                    "%lld) are interfering.", idx1, idx2, index_point[0], 
                    index_point[1], index_point[2], index_point[3], 
                    index_point[4], index_point[5], index_point[6],
                    get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 8
        case 8:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld,%lld,%lld,"
                    "%lld) of index space task %s (UID %lld) in parent task "
                    "%s (UID %lld) are interfering.", idx1, idx2, 
                    index_point[0], index_point[1], index_point[2], 
                    index_point[3], index_point[4], index_point[5], 
                    index_point[6], index_point[7], get_task_name(), 
                    get_unique_id(), parent_ctx->get_task_name(), 
                    parent_ctx->get_unique_id());
            break;
          }
#endif
#if LEGION_MAX_DIM >= 9
        case 9:
          {
            REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                    "Aliased and interfering region requirements for "
                    "point tasks are not permitted. Region requirements "
                    "%d and %d of point (%lld,%lld,%lld,%lld,%lld,%lld,%lld,"
                    "%lld,%lld) of index space task %s (UID %lld) in parent "
                    "task %s (UID %lld) are interfering.", idx1, idx2, 
                    index_point[0], index_point[1], index_point[2], 
                    index_point[3], index_point[4], index_point[5], 
                    index_point[6], index_point[7], index_point[8],
                    get_task_name(), get_unique_id(), 
                    parent_ctx->get_task_name(), parent_ctx->get_unique_id());
            break;
          }
#endif
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool PointTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      // Point tasks are never sent anywhere
      return true;
    }

    //--------------------------------------------------------------------------
    RtEvent PointTask::perform_mapping(MustEpochOp *must_epoch_owner/*=NULL*/,
                                       const DeferMappingArgs *args/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      // First time through do the concurrent analysis to get it in flight
      // and record any map applied conditions for it, note we'll always have
      // a non-null args for concurrent tasks so points can map in parallel
      if (concurrent_task && (args->invocation_count == 0))
      {
#ifdef DEBUG_LEGION
        assert(target_proc.exists());
#endif
        // If we're doing mapper checks then we need to do that now
        if (!runtime->unsafe_mapper)
        {
          const RtEvent checked = 
            slice_owner->verify_concurrent_execution(index_point, target_proc);
          if (checked.exists())
            map_applied_conditions.insert(checked);
        }
      }
      // For point tasks we use the point termination event which as the
      // end event for this task since point tasks can be moved and
      // the completion event is therefore not guaranteed to survive
      // the length of the task's execution
      const RtEvent deferred = map_all_regions(must_epoch_owner, args);
      if (deferred.exists())
        return deferred;
      slice_owner->record_point_mapped(mapped_event);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    bool PointTask::replicate_task(void)
    //--------------------------------------------------------------------------
    {
      // Pull this onto the stack since it is unsafe to read it after we
      // call the base class method
      SliceTask *owner = slice_owner;
      const RtEvent event = mapped_event;
      const bool result = SingleTask::replicate_task();
      if (result)
        owner->record_point_mapped(event);
      return result;
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_future_size(size_t return_type_size,
                   bool has_return_type_size, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      if (elide_future_return)
        return;
      if (has_return_type_size)
        slice_owner->handle_future_size(return_type_size,
                                        index_point, applied_events);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_output_extent(unsigned idx,
                            const DomainPoint &color, const DomainPoint &extent)
    //--------------------------------------------------------------------------
    {
      slice_owner->record_output_extent(idx, color, extent);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_output_registered(RtEvent registered,
                                             std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      slice_owner->record_output_registered(registered, applied_events);
    }

    //--------------------------------------------------------------------------
    void PointTask::shard_off(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
      slice_owner->record_point_mapped(mapped_precondition);
      SingleTask::shard_off(mapped_precondition);
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    VersionInfo& PointTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      // See if we've copied over the versions from our slice
      // if not we can just use our slice owner
      if (idx < version_infos.size())
        return version_infos[idx];
      return slice_owner->get_version_info(idx);
    }

    //--------------------------------------------------------------------------
    const VersionInfo& PointTask::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      // See if we've copied over the versions from our slice
      // if not we can just use our slice owner
      if (idx < version_infos.size())
        return version_infos[idx];
      return slice_owner->get_version_info(idx);
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_output_global(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return slice_owner->is_output_global(idx);
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_output_valid(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return slice_owner->is_output_valid(idx);
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind PointTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return POINT_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_TASK_COMPLETE_CALL);
      // Pass back our created and deleted operations 
      std::set<RtEvent> preconditions;
      if (execution_context != NULL)
      {
        slice_owner->return_privileges(execution_context, preconditions);
        // Invalidate any context that we had so that the child
        // operations can begin committing
        std::set<RtEvent> point_preconditions;
        execution_context->invalidate_region_tree_contexts(false,
                                            point_preconditions);
        if (!preconditions.empty())
          slice_owner->record_point_complete(
              Runtime::merge_events(preconditions));
        else
          slice_owner->record_point_complete(RtEvent::NO_RT_EVENT);
        if (runtime->legion_spy_enabled)
          execution_context->log_created_requirements();
        // See if we need to trigger that our children are complete
        const bool need_commit = execution_context->attempt_children_commit();
        // Mark that this operation is now complete
        if (!point_preconditions.empty())
          complete_operation(Runtime::merge_events(point_preconditions));
        else
          complete_operation();
        if (need_commit)
          trigger_children_committed();
      }
      else
      {
        if (!preconditions.empty())
          slice_owner->record_point_complete(
              Runtime::merge_events(preconditions));
        else
          slice_owner->record_point_complete(RtEvent::NO_RT_EVENT);
        complete_operation();
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_TASK_COMMIT_CALL);
      if (profiling_reported.exists())
        finalize_single_task_profiling();
      // A little strange here, but we don't directly commit this
      // operation, instead we just tell our slice that we are commited
      // In the deactivation of the slice task is when we will actually
      // have our commit call done
      slice_owner->record_point_committed(profiling_reported);
    }

    //--------------------------------------------------------------------------
    bool PointTask::pack_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_PACK_TASK_CALL);
      RezCheck z(rez);
      pack_single_task(rez, target);
      rez.serialize(orig_task);
#ifdef DEBUG_LEGION
      assert(is_origin_mapped()); // should be origin mapped if we're here
#endif
      // Return false since point tasks should always be deactivated
      // once they are sent to a remote node
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, POINT_UNPACK_TASK_CALL);
      DerezCheck z(derez);
      unpack_single_task(derez, ready_events);
      derez.deserialize(orig_task);
      set_current_proc(current);
      // Get the context information from our slice owner
      parent_ctx = slice_owner->get_context();
      parent_task = parent_ctx->get_task();
      set_provenance(slice_owner->get_provenance());
      // Remote point tasks are always resolved
      resolved = true;
      // We should always just apply these things now since we were mapped 
      // on the owner node
#ifdef DEBUG_LEGION
      assert(is_origin_mapped());
#endif
      // We're not going to get a callback from the context if we're a leaf
      if (!is_leaf())
      {
        Serializer rez;
        {
          RezCheck z2(rez);
          rez.serialize<SingleTask*>(orig_task);
          rez.serialize(mapped_event);
        }
        runtime->send_individual_remote_mapped(orig_proc, rez);
      }
      else
        complete_mapping();
      slice_owner->record_point_mapped(mapped_event);
      if (runtime->profiler != NULL)
        runtime->profiler->register_operation(this);
      return false;
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_post_execution(FutureInstance *instance,
                                  void *metadata, size_t metasize,
                                  FutureFunctor *functor, 
                                  Processor future_proc, bool own_functor)
    //--------------------------------------------------------------------------
    {
      if ((instance != NULL) && (instance->size > 0) && (shard_manager == NULL))
        check_future_return_bounds(instance);
      slice_owner->handle_future(single_task_termination, index_point,
          instance, metadata, metasize, functor, future_proc, own_functor); 
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_mispredication(void)
    //--------------------------------------------------------------------------
    {
      // First thing: increment the meta-task counts since we decremented
      // them in case we didn't end up running
#ifdef DEBUG_LEGION
      runtime->increment_total_outstanding_tasks(
          MispredicationTaskArgs::TASK_ID, true/*meta*/);
#else
      runtime->increment_total_outstanding_tasks();
#endif
#ifdef DEBUG_SHUTDOWN_HANG
      runtime->outstanding_counts[MispredicationTaskArgs::TASK_ID].fetch_add(1);
#endif
      slice_owner->set_predicate_false_result(index_point);
      // Pretend like we executed the task
      execution_context->handle_mispredication();
    }

    //--------------------------------------------------------------------------
    void PointTask::concurrent_allreduce(ProcessorManager *manager,
                           uint64_t lamport_clock, VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      slice_owner->concurrent_allreduce(this, manager, lamport_clock,
                                        vid, poisoned);
    }

    //--------------------------------------------------------------------------
    bool PointTask::check_concurrent_variant(VariantID vid)
    //--------------------------------------------------------------------------
    {
      if (vid == 0)
      {
        VariantImpl *impl = 
          runtime->find_variant_impl(task_id, selected_variant); 
        if (!impl->is_concurrent())
          return true;
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
          "Mapper %s selected a concurrent variant %d for point task %s "
          "(UID %lld) of a concurrent task launch but selected a "
          "non-concurrent variant for a different point task. All point "
          "tasks in a concurrent index task launch must be the same if "
          "any of them are going to be a concurrent variant.",
          mapper->get_mapper_name(), selected_variant, get_task_name(), 
          get_unique_id())
      }
      else if (vid != selected_variant)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
          "Mapper %s selected a concurrent variant %d for point task %s "
          "(UID %lld) of a concurrent task launch but selected a different "
          "concurrent variant %d for a different point task. All point "
          "tasks in a concurrent index task launch must use the same "
          "concurrent task variant.", mapper->get_mapper_name(),
          selected_variant, get_task_name(), get_unique_id(), vid)
      return true;
    }

    //--------------------------------------------------------------------------
    void PointTask::perform_concurrent_task_barrier(void)
    //--------------------------------------------------------------------------
    {
      // Check that this is a concurrent index space task launch
      if (!is_concurrent())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_CONCURRENT_TASK_BARRIER,
            "Illegal concurrent task barrier in task %s (UID %lld) which is "
            "not part of a concurrent index space task. Concurrent task "
            "barriers are only permitted in concurrent index space tasks.",
            get_task_name(), get_unique_id())
      if (!concurrent_task_barrier.exists())
      {
        concurrent_task_barrier = slice_owner->get_concurrent_task_barrier();
        if (!concurrent_task_barrier.exists())
          REPORT_LEGION_ERROR(ERROR_ILLEGAL_CONCURRENT_TASK_BARRIER,
            "Illegal concurrent task barrier in task %s (UID %lld) which is "
            "not a task variant that requested support for concurrent "
            "barriers. To request support you must mark the task variant "
            "as needing 'concurrent_barrier' support in the task variant "
            "registrar.", get_task_name(), get_unique_id())
      }
      Runtime::phase_barrier_arrive(concurrent_task_barrier, 1/*count*/);
      concurrent_task_barrier.wait();
      Runtime::advance_barrier(concurrent_task_barrier);
#ifdef DEBUG_LEGION
      // If you ever fail this assertion then we exhausted the number
      // of generations in a barrier. Hopefully CUDA will fix its bug
      // before we ever need to deal with this
      assert(concurrent_task_barrier.exists());
#endif
    }

    //--------------------------------------------------------------------------
    const DomainPoint& PointTask::get_domain_point(void) const
    //--------------------------------------------------------------------------
    {
      return index_point;
    }

    //--------------------------------------------------------------------------
    void PointTask::set_projection_result(unsigned idx, LogicalRegion result)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < get_region_count());
#endif
      RegionRequirement &req = logical_regions[idx];
#ifdef DEBUG_LEGION
      assert(req.handle_type != LEGION_SINGULAR_PROJECTION);
#endif
      req.region = result;
      req.handle_type = LEGION_SINGULAR_PROJECTION;
      // Check to see if the region is a NO_REGION,
      // if it is then switch the privilege to NO_ACCESS
      if (req.region == LogicalRegion::NO_REGION)
        req.privilege = LEGION_NO_ACCESS;
    }

    //--------------------------------------------------------------------------
    void PointTask::initialize_point(SliceTask *owner, const DomainPoint &point,
                                    const FutureMap &point_arguments,bool eager,
                                    const std::vector<FutureMap> &point_futures)
    //--------------------------------------------------------------------------
    {
      slice_owner = owner;
      // Get our point
      index_point = point;
      // Get our argument
      if (point_arguments.impl != NULL)
      {
        Future f = point_arguments.impl->get_future(point, true/*internal*/);
        if (f.impl != NULL)
        {
          // Request the local buffer
          f.impl->request_runtime_instance(this, eager);
          // Make sure that it is ready
          const RtEvent ready = f.impl->subscribe();
          if (ready.exists() && !ready.has_triggered())
            ready.wait();
          const void *buffer =
            f.impl->find_runtime_buffer(parent_ctx, local_arglen);
          // Have to make a local copy since the point takes ownership
          if (local_arglen > 0)
          {
            local_args = malloc(local_arglen);
            memcpy(local_args, buffer, local_arglen); 
          }
        }
      }
      if (!point_futures.empty())
      {
        for (std::vector<FutureMap>::const_iterator it = 
              point_futures.begin(); it != point_futures.end(); it++)
          this->futures.push_back(it->impl->get_future(point,true/*internal*/));
      }
    }

    //--------------------------------------------------------------------------
    void PointTask::complete_replay(ApEvent instance_ready_event,
                                    ApEvent completion_postcondition)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_leaf());
      assert(is_origin_mapped());
      assert(!target_processors.empty());
      assert(single_task_termination.exists());
      assert(region_preconditions.empty());
#endif
      if (completion_postcondition.exists())
        record_completion_effect(completion_postcondition);
      const AddressSpaceID target_space =
        runtime->find_address_space(target_processors.front());
      // Check to see if we're replaying this locally or remotely
      if (target_space != runtime->address_space)
      {
        // This is the remote case, pack it up and ship it over
        // Update our target_proc so that the sending code is correct 
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(instance_ready_event);
          rez.serialize(completion_postcondition);
          rez.serialize(target_processors.front());
          rez.serialize(SLICE_TASK_KIND);
          slice_owner->pack_task(rez, target_space);
        }
        runtime->send_remote_task_replay(target_space, rez);
      }
      else
      {
        // This is the local case
        region_preconditions.resize(regions.size(), instance_ready_event);
        execution_fence_event = instance_ready_event;
        update_no_access_regions();
        launch_task();
      }
    }

    //--------------------------------------------------------------------------
    TraceLocalID PointTask::get_trace_local_id(void) const
    //--------------------------------------------------------------------------
    {
      return TraceLocalID(trace_local_id, get_domain_point());
    }

    //--------------------------------------------------------------------------
    size_t PointTask::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return slice_owner->get_collective_points();
    }

    //--------------------------------------------------------------------------
    bool PointTask::find_shard_participants(std::vector<ShardID> &shards)
    //--------------------------------------------------------------------------
    {
      return slice_owner->find_shard_participants(shards);
    }

    //--------------------------------------------------------------------------
    RtEvent PointTask::convert_collective_views(unsigned requirement_index,
                       unsigned analysis_index, LogicalRegion region,
                       const InstanceSet &targets, InnerContext *physical_ctx,
                       CollectiveMapping *&analysis_mapping, bool &first_local,
                       LegionVector<FieldMaskSet<InstanceView> > &target_views,
                       std::map<InstanceView*,size_t> &collective_arrivals)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled)
        LegionSpy::log_collective_rendezvous(unique_op_id, 
                        requirement_index, analysis_index);
      return slice_owner->convert_collective_views(requirement_index, 
          analysis_index, region, targets, physical_ctx, analysis_mapping,
          first_local, target_views, collective_arrivals);
    }

    //--------------------------------------------------------------------------
    RtEvent PointTask::perform_collective_versioning_analysis(unsigned index,
        LogicalRegion handle, EqSetTracker *tracker, const FieldMask &mask,
        unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return slice_owner->perform_collective_versioning_analysis(index,
          handle, tracker, mask, parent_req_index);
    }

    //--------------------------------------------------------------------------
    void PointTask::perform_replicate_collective_versioning(unsigned index,
        unsigned parent_req_index, LegionMap<LogicalRegion,
            CollectiveVersioningBase::RegionVersioning> &to_perform)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_manager != NULL);
#endif
      if (IS_COLLECTIVE(regions[index]) || std::binary_search(
            check_collective_regions.begin(), 
            check_collective_regions.end(), index))
        slice_owner->perform_replicate_collective_versioning(index,
            parent_req_index, to_perform);
      else
        SingleTask::perform_replicate_collective_versioning(index,
            parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    void PointTask::convert_replicate_collective_views(
          const CollectiveViewCreatorBase::RendezvousKey &key,
          std::map<LogicalRegion,
            CollectiveViewCreatorBase::CollectiveRendezvous> &rendezvous)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_manager != NULL);
#endif
      if (IS_COLLECTIVE(regions[key.region_index]) || std::binary_search(
            check_collective_regions.begin(), 
            check_collective_regions.end(), key.region_index))
        slice_owner->convert_replicate_collective_views(key, rendezvous);
      else
        SingleTask::convert_replicate_collective_views(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_intra_space_dependences(unsigned index,
                                    const std::vector<DomainPoint> &dependences)
    //--------------------------------------------------------------------------
    {
      if (concurrent_task)
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_CONCURRENT_EXECUTION,
            "Concurrent index space task %s (UID %lld) has intra-index-space "
            "dependences on region requirement %d. It is illegal to have "
            "intra-index-space dependences on concurrent executions because "
            "the resulting execution is guaranteed to hang.", 
            get_task_name(), get_unique_id(), index)
      if (!check_collective_regions.empty())
      {
        if (mapper == NULL)
          mapper = runtime->find_mapper(current_proc, map_id);
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
            "Mapper %s asked for collective region checks for index task "
            "%s (UID %lld) but this task has intra-index-space task "
            "dependences. Collective behavior cannot be analyzed on task "
            "with inter-index-space dependences.", mapper->get_mapper_name(),
            get_task_name(), get_unique_id())
      }
      // Scan through the list until we find ourself
      for (unsigned idx = 0; idx < dependences.size(); idx++)
      {
        if (dependences[idx] == index_point)
        {
          // If we've got a prior dependence then record it
          if (idx > 0)
          {
            const DomainPoint &prev = dependences[idx-1];
            const RtEvent pre = slice_owner->find_intra_space_dependence(prev);
            if (!std::binary_search(intra_space_mapping_dependences.begin(),
                  intra_space_mapping_dependences.end(), pre))
            {
              intra_space_mapping_dependences.push_back(pre);
              std::sort(intra_space_mapping_dependences.begin(),
                        intra_space_mapping_dependences.end());
            }
            if (runtime->legion_spy_enabled)
            {
              // We know we only need a dependence on the previous point but
              // Legion Spy is stupid, so log everything we have a
              // precondition on even if it is transitively implied
              for (unsigned idx2 = 0; idx2 < idx; idx2++)
                LegionSpy::log_intra_space_dependence(unique_op_id,
                                                      dependences[idx2]);
            }
          }
          // If we're not the last dependence, then send our mapping event
          // so that others can record a dependence on us
          if (idx < (dependences.size()-1))
            slice_owner->record_intra_space_dependence(index_point,
                                                       dependences[idx+1], 
                                                       get_mapped_event());
          return;
        }
      }
      // We should never get here
      assert(false);
    }

    /////////////////////////////////////////////////////////////
    // Shard Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardTask::ShardTask(Runtime *rt, SingleTask *source, InnerContext *parent,
        ShardManager *manager, ShardID id, Processor proc, VariantID variant)
      : SingleTask(rt), shard_id(id), all_shards_complete(false)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(proc.address_space() == runtime->address_space);
#endif
      SingleTask::activate();
      set_current_proc(proc); // do this before clone_single_from
      if (source != NULL)
        clone_single_from(source);
      else
        parent_ctx = parent;
      resolved = true;
      stealable = false;
      replicate = false;
      shard_manager = manager;
      shard_manager->add_base_gc_ref(SINGLE_TASK_REF);
      selected_variant = variant;
      shard_barrier = shard_manager->get_shard_task_barrier();
      // If we have any region requirements then they are all collective
      check_collective_regions.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        check_collective_regions[idx] = idx;
      if (runtime->legion_spy_enabled)
      {
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
          log_requirement(unique_op_id, idx, logical_regions[idx]);
      }
    }

    //--------------------------------------------------------------------------
    ShardTask::ShardTask(Runtime *rt, InnerContext *parent, Deserializer &derez,
        ShardManager *manager, ShardID id, Processor proc, VariantID variant)
      : SingleTask(rt), shard_id(id), all_shards_complete(false)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(proc.address_space() == runtime->address_space);
#endif
      SingleTask::activate();
      set_current_proc(proc);
      std::set<RtEvent> ready_events;
      unpack_single_task(derez, ready_events);
      resolved = true;
      stealable = false;
      replicate = false;
      parent_ctx = parent;
      shard_manager = manager;
      shard_manager->add_base_gc_ref(SINGLE_TASK_REF);
      selected_variant = variant;
      shard_barrier = shard_manager->get_shard_task_barrier();
      // If we have any region requirements then they are all collective
      check_collective_regions.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        check_collective_regions[idx] = idx;
      if (runtime->legion_spy_enabled)
      {
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
          log_requirement(unique_op_id, idx, logical_regions[idx]);
      }
      if (!ready_events.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(ready_events);
        wait_on.wait();
      }
    }
    
    //--------------------------------------------------------------------------
    ShardTask::~ShardTask(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_manager == NULL);
#endif
    }

    //--------------------------------------------------------------------------
    void ShardTask::activate(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Set our shard manager to NULL since we are not supposed to delete it
      if (shard_manager->remove_base_gc_ref(SINGLE_TASK_REF))
        delete shard_manager;
      shard_manager = NULL;
      SingleTask::deactivate(false/*free*/);
    }

    //--------------------------------------------------------------------------
    Domain ShardTask::get_slice_domain(void) const
    //--------------------------------------------------------------------------
    {
      // Shards have already been sliced down to single points
      return Domain(index_point, index_point);
    }

    //--------------------------------------------------------------------------
    size_t ShardTask::get_total_shards(void) const
    //--------------------------------------------------------------------------
    {
      return shard_manager->total_shards;
    }

    //--------------------------------------------------------------------------
    DomainPoint ShardTask::get_shard_point(void) const
    //--------------------------------------------------------------------------
    {
      return shard_manager->shard_points[shard_id];
    }

    //--------------------------------------------------------------------------
    Domain ShardTask::get_shard_domain(void) const
    //--------------------------------------------------------------------------
    {
      return shard_manager->shard_domain;
    }

    //--------------------------------------------------------------------------
    bool ShardTask::is_top_level_task(void) const
    //--------------------------------------------------------------------------
    {
      return shard_manager->top_level_task;
    }

    //--------------------------------------------------------------------------
    void ShardTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool ShardTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    RtEvent ShardTask::perform_must_epoch_version_analysis(MustEpochOp *own)
    //--------------------------------------------------------------------------
    {
      assert(false);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    RtEvent ShardTask::perform_mapping(MustEpochOp *must_epoch_owner, 
                                       const DeferMappingArgs *args)
    //--------------------------------------------------------------------------
    {
      const RtEvent deferred = map_all_regions(must_epoch_owner, args);
      if (deferred.exists())
        return deferred;
      shard_manager->handle_post_mapped(true/*local*/, mapped_event);
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_future_size(size_t return_type_size,
                   bool has_return_type_size, std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
      // do nothing 
    }
    
    //--------------------------------------------------------------------------
    bool ShardTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return false;
    }

    //--------------------------------------------------------------------------
    void ShardTask::initialize_map_task_input(Mapper::MapTaskInput &input,
                                              Mapper::MapTaskOutput &output,
                                              MustEpochOp *must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      SingleTask::initialize_map_task_input(input, output, must_epoch_owner);
      input.shard = get_shard_point(); 
      input.shard_domain = get_shard_domain();
      input.shard_processor = current_proc; 
      input.shard_variant = selected_variant;
      output.chosen_variant = selected_variant;
      output.target_procs.resize(1, current_proc);
    }

    //--------------------------------------------------------------------------
    void ShardTask::finalize_map_task_output(Mapper::MapTaskInput &input,
                                             Mapper::MapTaskOutput &output,
                                             MustEpochOp *must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      // This is a replicated task, the mapper isn't allowed to 
      // mutate the target_processors from the shard processor
      if ((output.target_procs.size() != 1) ||
          (output.target_procs.front() != input.shard_processor))
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
            "Mapper %s provided invalid target_processors from call to "
            "'map_task' for replicated task %s (UID %lld). Replicated "
            "tasks are only permitted to have one target processor and "
            "it must be exactly 'input.shard_procesor' as that is where "
            "this replicated copy of the task has been assigned to run "
            "by this same mapper.", mapper->get_mapper_name(),
            get_task_name(), get_unique_id())
      if (output.chosen_variant != input.shard_variant)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output from invocation of '%s' on "
                      "mapper %s. Mapper specified an invalid task variant "
                      "of ID %d for replicated task %s (ID %lld), which "
                      "differs from the specified 'input.shard_variant' %d "
                      "previously chosen by the mapper in 'replicate_task'. "
                      "The mapper is required to maintain the previously "
                      "selected variant in the output 'map_task'.",
                      "map_task", mapper->get_mapper_name(),
                      output.chosen_variant, get_task_name(),
                      get_unique_id(), input.shard_variant)
      SingleTask::finalize_map_task_output(input, output, must_epoch_owner);
      if (!is_leaf() && !regions.empty() && !runtime->unsafe_mapper)
      {
#ifdef DEBUG_LEGION
        assert(mapper != NULL);
        assert(regions.size() == virtual_mapped.size());
#endif
        // If this is not a leaf shard then check that all the shards agree
        // on which regions are going to be virtually mapped and which aren't
        shard_manager->rendezvous_check_virtual_mappings(shard_id, mapper,
                                                         virtual_mapped);
      }
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind ShardTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return SHARD_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void ShardTask::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_barrier.exists() == shard_manager->control_replicated);
#endif
      // We need to ensure that each shard has gotten this call to ensure that
      // all the child operations are done and have propagated all their context
      // information back to us before we go about invalidating our contexts
      if (!all_shards_complete && shard_manager->control_replicated)
      {
        Runtime::phase_barrier_arrive(shard_barrier, 1/*false*/);
        const RtEvent shards_complete = shard_barrier;
        Runtime::advance_barrier(shard_barrier);
        all_shards_complete = true;
        if (!shards_complete.has_triggered())
        {
          DeferTriggerTaskCompleteArgs args(this);
          runtime->issue_runtime_meta_task(args,
              LG_LATENCY_DEFERRED_PRIORITY, shards_complete);
          return;
        }
      }
      // First do the normal clean-up operations
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((outstanding_profiling_requests.fetch_sub(1) == 1) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
      // Invalidate any context that we had so that the child
      // operations can begin committing
      std::set<RtEvent> preconditions;
      execution_context->invalidate_region_tree_contexts(is_top_level_task(),
          preconditions, &shard_manager->get_mapping(), shard_id);
      if (runtime->legion_spy_enabled)
        execution_context->log_created_requirements();
      ApEvent task_effects = single_task_termination;
      if (!task_completion_effects.empty())
      {
        task_completion_effects.insert(single_task_termination);
        task_effects = Runtime::merge_events(NULL, task_completion_effects);
      }
      const RtEvent shard_event =
        shard_manager->trigger_task_complete(true/*local*/, task_effects);
      if (shard_event.exists())
        preconditions.insert(shard_event);
      // See if we need to trigger that our children are complete
      const bool need_commit = execution_context->attempt_children_commit();
      // Make sure all the shards are complete together
      if (shard_manager->control_replicated)
      {
        if (!preconditions.empty())
          Runtime::phase_barrier_arrive(shard_barrier, 1/*count*/,
              Runtime::merge_events(preconditions));
        else
          Runtime::phase_barrier_arrive(shard_barrier, 1/*count*/);
        complete_operation(shard_barrier);
      }
      else
        complete_operation();
      if (need_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void ShardTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      // Commit this operation
      // Dont' deactivate ourselves, the shard manager will do that for us
      commit_operation(false/*deactivate*/, profiling_reported);
      // If we still have to report profiling information then we must
      // block here to avoid a race with the shard manager deactivating
      // us before we are done with this object
      if (profiling_reported.exists() && !profiling_reported.has_triggered())
        profiling_reported.wait();
      // Lastly invoke the method on the shard manager, this could
      // delete us so it has to be last
      shard_manager->trigger_task_commit(true/*local*/);
    }

    //--------------------------------------------------------------------------
    bool ShardTask::pack_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_single_task(rez, target);
      parent_ctx->pack_inner_context(rez);
      return false;
    }

    //--------------------------------------------------------------------------
    bool ShardTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
#ifdef DEBUG_LEGION
      assert(!single_task_termination.exists());
#endif
      unpack_single_task(derez, ready_events);
      parent_ctx = InnerContext::unpack_inner_context(derez, runtime);
      set_current_proc(current);
      return false;
    }

    //--------------------------------------------------------------------------
    void ShardTask::perform_inlining(VariantImpl *variant,
                                  const std::deque<InstanceSet> &parent_regions)
    //--------------------------------------------------------------------------
    {
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_post_execution(FutureInstance *instance,
        void *metadata, size_t metasize,
        FutureFunctor *functor, Processor future_proc, bool own_functor)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(functor == NULL);
#endif
      if ((instance != NULL) && (instance->size > 0))
        check_future_return_bounds(instance);
      shard_manager->handle_post_execution(instance, single_task_termination,
                                           metadata, metasize, true/*local*/);
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_mispredication(void)
    //--------------------------------------------------------------------------
    {
      // TODO: figure out how mispredication works with control replication
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardTask::concurrent_allreduce(ProcessorManager *manager,
                           uint64_t lamport_clock, VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void ShardTask::perform_concurrent_task_barrier(void)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_ILLEGAL_CONCURRENT_TASK_BARRIER,
          "Illegal concurrent task barrier performed in replicated task %s "
          "(UID %lld). Concurrent task barriers are not permitted in "
          "replicated tasks. They can only be performed in concurrent index "
          "space tasks.", get_task_name(), get_unique_id())
    }

    //--------------------------------------------------------------------------
    RtEvent ShardTask::convert_collective_views(unsigned requirement_index,
                       unsigned analysis_index, LogicalRegion region,
                       const InstanceSet &targets, InnerContext *physical_ctx,
                       CollectiveMapping *&analysis_mapping, bool &first_local,
                       LegionVector<FieldMaskSet<InstanceView> > &target_views,
                       std::map<InstanceView*,size_t> &collective_arrivals)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled)
        LegionSpy::log_collective_rendezvous(unique_op_id, 
                        requirement_index, analysis_index);
      return shard_manager->convert_collective_views(requirement_index,
          analysis_index, region, targets, physical_ctx, analysis_mapping,
          first_local, target_views, collective_arrivals);
    }

    //--------------------------------------------------------------------------
    RtEvent ShardTask::perform_collective_versioning_analysis(unsigned index,
        LogicalRegion handle, EqSetTracker *tracker, const FieldMask &mask,
        unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return shard_manager->rendezvous_collective_versioning_analysis(index,
          handle, tracker, runtime->address_space, mask, parent_req_index);
    }

    //--------------------------------------------------------------------------
    TaskContext* ShardTask::create_execution_context(VariantImpl *v,
        std::set<ApEvent> &launch_events, bool inline_task, bool leaf_task)
    //--------------------------------------------------------------------------
    {
      if (runtime->legion_spy_enabled)
        LegionSpy::log_shard(LEGION_DISTRIBUTED_ID_FILTER(shard_manager->did),
                             shard_id, get_unique_id());
      if (!leaf_task)
      {
        // If we have a control replication context then we do the special path
        ReplicateContext *repl_ctx = new ReplicateContext(runtime, this,
            get_depth(), v->is_inner(), regions, output_regions,
            parent_req_indexes, virtual_mapped, execution_fence_event,
            shard_manager, inline_task, parent_ctx->is_concurrent_context());
        repl_ctx->add_base_gc_ref(SINGLE_TASK_REF);
        if (mapper == NULL)
          mapper = runtime->find_mapper(current_proc, map_id);
        repl_ctx->configure_context(mapper, task_priority);
        // Save the execution context early since we'll need it
        execution_context = repl_ctx;
        // Make sure that none of the shards start until all the replicate
        // contexts have been made across all the shards
        RtEvent ready = complete_startup_initialization();
        launch_events.insert(ApEvent(ready));
      }
      else
      {
        execution_context = new LeafContext(runtime, this, inline_task);
        execution_context->add_base_gc_ref(SINGLE_TASK_REF);
      }
      return execution_context;
    }

    //--------------------------------------------------------------------------
    InnerContext* ShardTask::create_implicit_context(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext *repl_ctx = new ReplicateContext(runtime, this,
          get_depth(), false/*is inner*/, regions, output_regions,
          parent_req_indexes, virtual_mapped, execution_fence_event,
          shard_manager, false/*inline task*/, true/*implicit*/);
      repl_ctx->add_base_gc_ref(SINGLE_TASK_REF);
      // Save the execution context early since we'll need it
      execution_context = repl_ctx;
      // Wait until all the other shards are ready too
      const RtEvent wait_on = complete_startup_initialization();
      if (!wait_on.has_triggered())
        wait_on.wait();
      return repl_ctx;
    }

    //--------------------------------------------------------------------------
    RtEvent ShardTask::complete_startup_initialization(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(shard_manager->control_replicated == shard_barrier.exists());
#endif
      // We only do this for control replicated tasks
      if (shard_manager->control_replicated)
      {
        Runtime::phase_barrier_arrive(shard_barrier, 1/*count*/);
        const RtEvent result = shard_barrier;
        // Advance this for when we get to completion
        Runtime::advance_barrier(shard_barrier);
        return result;
      }
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void ShardTask::dispatch(void)
    //--------------------------------------------------------------------------
    {
      // Have to launch a task to do this in case they need to rendezvous
      const RtUserEvent shard_mapped = Runtime::create_rt_user_event();
      DeferMappingArgs args(this, NULL, shard_mapped,
          0/*invocation count*/, NULL/*performed*/, NULL/*effects*/);
      runtime->issue_runtime_meta_task(args, LG_THROUGHPUT_DEFERRED_PRIORITY);
      // Then defer the launching of the shard when the mapping is done
      defer_launch_task(shard_mapped);
    }

    //--------------------------------------------------------------------------
    void ShardTask::return_resources(ResourceTracker *target,
                                     std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(execution_context != NULL);
#endif
      execution_context->return_resources(target, context_index, preconditions);
    }

    //--------------------------------------------------------------------------
    void ShardTask::report_leaks_and_duplicates(
                                               std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(execution_context != NULL);
#endif
      execution_context->report_leaks_and_duplicates(preconditions);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_collective_message(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      get_replicate_context()->handle_collective_message(derez);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_rendezvous_message(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      get_replicate_context()->handle_rendezvous_message(derez);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_compute_equivalence_sets(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      get_replicate_context()->handle_compute_equivalence_sets(derez);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_output_equivalence_set(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      get_replicate_context()->handle_output_equivalence_set(derez);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_refine_equivalence_sets(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      get_replicate_context()->handle_refine_equivalence_sets(derez);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_intra_space_dependence(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      get_replicate_context()->handle_intra_space_dependence(derez);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_resource_update(Deserializer &derez,
                                           std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      get_replicate_context()->handle_resource_update(derez, applied);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_created_region_contexts(Deserializer &derez,
                                                   std::set<RtEvent> &applied)
    //--------------------------------------------------------------------------
    {
      get_replicate_context()->handle_created_region_contexts(derez, applied);
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_trace_update(Deserializer &derez, 
                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      get_replicate_context()->handle_trace_update(derez, source);
    }

    //--------------------------------------------------------------------------
    ApBarrier ShardTask::handle_find_trace_shard_event(size_t template_index,
                                            ApEvent event, ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
      return get_replicate_context()->handle_find_trace_shard_event(
          template_index, event, remote_shard);
    }

    //--------------------------------------------------------------------------
    ApBarrier ShardTask::handle_find_trace_shard_frontier(size_t template_index,
                                            ApEvent event, ShardID remote_shard)
    //--------------------------------------------------------------------------
    {
      return get_replicate_context()->handle_find_trace_shard_frontier(
          template_index, event, remote_shard);
    }

    //--------------------------------------------------------------------------
    ReplicateContext* ShardTask::get_replicate_context(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(execution_context != NULL);
      ReplicateContext *repl_ctx = 
        dynamic_cast<ReplicateContext*>(execution_context);
      assert(repl_ctx != NULL);
      return repl_ctx;
#else
      return static_cast<ReplicateContext*>(execution_context);
#endif
    }

    //--------------------------------------------------------------------------
    void ShardTask::initialize_implicit_task(TaskID tid,
                                             MapperID mid, Processor proxy)
    //--------------------------------------------------------------------------
    {
      task_id = tid;
      map_id = mid;
      orig_proc = proxy;
      current_proc = proxy;
      shard_manager->handle_post_mapped(true/*local*/, mapped_event);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_completion_effect(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      if (!effect.exists())
        return;
      // If we're recording then we also need to capture this for later
      if (is_recording())
      {
        AutoLock o_lock(op_lock);
        task_completion_effects.insert(effect);
      }
      slice_owner->record_completion_effect(effect);
    } 
   
    //--------------------------------------------------------------------------
    void PointTask::record_completion_effect(ApEvent effect,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      if (!effect.exists())
        return;
      // If we're recording then we also need to capture this for later
      if (is_recording())
      {
        AutoLock o_lock(op_lock);
        task_completion_effects.insert(effect);
      }
      slice_owner->record_completion_effect(effect, map_applied_events);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_completion_effects(const std::set<ApEvent> &effects)
    //--------------------------------------------------------------------------
    {
      if (effects.empty())
        return;
      // If we're recording then we also need to capture this for later
      if (is_recording())
      {
        AutoLock o_lock(op_lock);
        for (std::set<ApEvent>::const_iterator it =
              effects.begin(); it != effects.end(); it++)
          if (it->exists())
            task_completion_effects.insert(*it);
      }
      slice_owner->record_completion_effects(effects);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_completion_effects(
                                            const std::vector<ApEvent> &effects)
    //--------------------------------------------------------------------------
    {
      if (effects.empty())
        return;
      // If we're recording then we also need to capture this for later
      if (is_recording())
      {
        AutoLock o_lock(op_lock);
        for (std::vector<ApEvent>::const_iterator it =
              effects.begin(); it != effects.end(); it++)
          if (it->exists())
            task_completion_effects.insert(*it);
      }
      slice_owner->record_completion_effects(effects);
    }

    //--------------------------------------------------------------------------
    bool PointTask::has_remaining_inlining_dependences(
                   std::map<PointTask*,unsigned> &remaining,
                   std::map<RtEvent,std::vector<PointTask*> > &event_deps) const
    //--------------------------------------------------------------------------
    {
      if (intra_space_mapping_dependences.empty())
        return false;
      unsigned count = 0;
      for (std::vector<RtEvent>::const_iterator it =
            intra_space_mapping_dependences.begin(); it !=
            intra_space_mapping_dependences.end(); it++)
      {
        if (it->has_triggered())
          continue;
        count++;
        event_deps[*it].push_back(const_cast<PointTask*>(this));
      }
      if (count > 0)
      {
        remaining[const_cast<PointTask*>(this)] = count;
        return true;
      }
      else
        return false;
    }

    /////////////////////////////////////////////////////////////
    // Index Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(Runtime *rt)
      : MultiTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(const IndexTask &rhs)
      : MultiTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    IndexTask::~IndexTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    IndexTask& IndexTask::operator=(const IndexTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void IndexTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_ACTIVATE_CALL);
      MultiTask::activate();
      serdez_redop_fns = NULL;
      total_points = 0;
      mapped_points = 0;
      complete_points = 0;
      committed_points = 0;
      concurrent_points = 0;
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      copy_fill_priority = 0;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
    }

    //--------------------------------------------------------------------------
    void IndexTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_DEACTIVATE_CALL);
      reduction_instance = NULL; // we don't own this so clear it
      MultiTask::deactivate(false/*free*/);
      if (!origin_mapped_slices.empty())
      {
        for (std::set<SliceTask*>::const_iterator it = 
              origin_mapped_slices.begin(); it != 
              origin_mapped_slices.end(); it++)
        {
          (*it)->deactivate();
        }
        origin_mapped_slices.clear();
      }
      if (!reduction_instances.empty())
      {
        for (std::vector<FutureInstance*>::const_iterator it =
              reduction_instances.begin(); it != 
              reduction_instances.end(); it++)
          delete (*it);
        reduction_instances.clear();
      }
      serdez_redop_targets.clear();
      // Remove our reference to the reduction future
      reduction_future = Future();
      map_applied_conditions.clear();
      output_preconditions.clear();
      complete_preconditions.clear();
      commit_preconditions.clear();
      version_infos.clear();
      if (!profiling_info.empty())
      {
        for (unsigned idx = 0; idx < profiling_info.size(); idx++)
          free(profiling_info[idx].buffer);
        profiling_info.clear();
      }
      interfering_requirements.clear();
      point_requirements.clear();
      concurrent_slices.clear();
      if (concurrent_task_barrier.exists())
        concurrent_task_barrier.destroy_barrier();
#ifdef DEBUG_LEGION
      assert(pending_intra_space_dependences.empty());
#endif
      if (freeop)
        runtime->free_index_task(this);
    }

    //--------------------------------------------------------------------------
    void IndexTask::validate_output_extents(unsigned index,
                                            const OutputRequirement& req,
                                    const OutputExtentMap& output_extents) const
    //--------------------------------------------------------------------------
    {
      size_t num_tasks = 0;
      if (sharding_space.exists())
        num_tasks = runtime->forest->get_domain_volume(sharding_space);
      else
        num_tasks = launch_space->get_volume();

      if (output_extents.size() == num_tasks) return;

      REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_PROJECTION,
        "A projection functor for every output requirement must be "
        "bijective, but projection functor %u for output requirement %u "
        "in task %s (UID: %lld) mapped more than one point in the launch "
        "domain to the same subregion.",
        req.projection, index, get_task_name(), get_unique_op_id());
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_output_extents(
                                   std::vector<OutputExtentMap> &output_extents)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(output_region_extents.size() == output_extents.size());
#endif
      {
        AutoLock o_lock(op_lock);
        for (unsigned idx = 0; idx < output_extents.size(); idx++)
        {
          OutputExtentMap &target = output_region_extents[idx];
          if (!target.empty())
          {
            // Merge the new extents in
            OutputExtentMap &extents = output_extents[idx];
            for (OutputExtentMap::const_iterator it =
                  extents.begin(); it != extents.end(); it++)
            {
              if (target.find(it->first) != target.end()) 
              {
                const DomainPoint& color = it->first;
                const OutputRequirement& req = output_regions[idx];
                std::stringstream ss;
                ss << "(" << color[0];
                for (int dim = 1; dim < color.dim; ++dim)
                  ss << "," << color[dim];
                ss << ")";
                REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_PROJECTION,
                  "A projection functor for every output requirement must be "
                  "bijective, but projection functor %u for output requirement "
                  "%u in task %s (UID: %lld) mapped more than one point "
                  "in the launch domain to the same subregion of color %s.",
                  req.projection, idx, get_task_name(), get_unique_op_id(),
                  ss.str().c_str());
              }
              target.insert(*it);
            }
          }
          else
            target.swap(output_extents[idx]);
        }
        // Now Check to see if we've received all the extents
        for (unsigned idx = 0; idx < output_region_extents.size(); idx++)
        {
          if (is_output_valid(idx))
            continue;
#ifdef DEBUG_LEGION
          assert(output_region_extents[idx].size() <= total_points);
#endif
          if (output_region_extents[idx].size() < total_points)
            return;
        }
      }
      // If we get here then we can finalize our output regions
      finalize_output_regions(true/*first invocation*/);
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_output_registered(RtEvent registered)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(registered.exists());
      assert(!output_regions.empty());
#endif
      // Record it in the set of output events and if we've seen all of them
      // then we can launch the meta-task to do the final regisration
      AutoLock o_lock(op_lock);
      output_preconditions.push_back(registered);
#ifdef DEBUG_LEGION
      assert(output_preconditions.size() <= total_points);
#endif
      if (output_preconditions.size() == total_points)
      {
        // Can only mark the EqKDTree ready once all the points are registered
        FinalizeOutputEqKDTreeArgs args(this);
        registered = runtime->issue_runtime_meta_task(args,
            LG_LATENCY_DEFERRED_PRIORITY,
            Runtime::merge_events(output_preconditions));
        complete_preconditions.insert(registered);
      }
    }

    //--------------------------------------------------------------------------
    Domain IndexTask::compute_global_output_ranges(IndexSpaceNode *parent,
                                                   IndexPartNode *part,
                                          const OutputExtentMap& output_extents,
                                          const OutputExtentMap& local_extents)
    //--------------------------------------------------------------------------
    {
      // First, we collect all the extents of local outputs.
      // While doing this, we also check the alignment.
      Domain color_space;
      part->color_space->get_domain(color_space);
#ifdef DEBUG_LEGION
      assert(color_space.dense());
#endif
      int32_t ndim = color_space.dim;
      DomainPoint color_extents = color_space.hi() - color_space.lo() + 1;

#ifdef DEBUG_LEGION
      // Check alignments between tiles
      for (OutputExtentMap::const_iterator it = output_extents.begin();
           it != output_extents.end(); ++it)
      {
        const DomainPoint &color = it->first;
        const DomainPoint &extent = it->second;

        for (int32_t dim = 0; dim < ndim; ++dim)
        {
          if (color[dim] == 0) continue;
          DomainPoint neighbor = color;
          --neighbor[dim];
          auto finder = output_extents.find(neighbor);
          assert(finder != output_extents.end());

          const DomainPoint &neighbor_extent = it->second;
          if (extent[dim] != neighbor_extent[dim])
          {
              std::stringstream ss;
              ss << "Point task " << color << " returned an output of extent "
                 << extent[dim] << " for dimension " << dim
                 << ", but an adjacent point task returned an output of extent "
                 << neighbor_extent[dim] << ". "
                 << "Please make sure the outputs from point tasks are aligned.";
              REPORT_LEGION_ERROR(
                  ERROR_UNALIGNED_OUTPUT_REGION, "%s", ss.str().c_str());
          }
        }
      }
#endif

      // Initialize the vectors of extents with 0
      std::vector<std::vector<coord_t>> all_extents(ndim);
      for (int32_t dim = 0; dim < ndim; ++dim)
        all_extents[dim].resize(color_extents[dim] + 1, 0);

      // Populate the extent vectors
      for (OutputExtentMap::const_iterator it = output_extents.begin();
           it != output_extents.end(); ++it)
      {
        const DomainPoint &color = it->first;
        const DomainPoint &extent = it->second;
        for (int32_t dim = 0; dim < ndim; ++dim)
        {
          coord_t c = color[dim];
          coord_t ext = extent[dim];
          coord_t &to_update = all_extents[dim][c];
          // Ignore all zero extents when populating the extent vector
          if (to_update == 0 && ext > 0) to_update = ext;
        }
      }

      // Prefix sum the extents to get sub-ranges for each dimension
      for (int32_t dim = 0; dim < ndim; ++dim) {
        std::vector<coord_t> &extents = all_extents[dim];
        coord_t sum = 0;
        for (size_t idx = 0; idx < extents.size() - 1; ++idx)
        {
          coord_t ext = extents[idx];
          extents[idx] = sum;
          sum += ext;
        }
        extents.back() = sum;
      }

      // Initialize the subspaces using the compute sub-ranges
      for (OutputExtentMap::const_iterator it = output_extents.begin();
           it != output_extents.end(); ++it)
      {
        const DomainPoint &color = it->first;

        // If this subspace isn't local to us, we are not allowed to
        // set its range.
        if (local_extents.find(color) == local_extents.end()) continue;

        IndexSpaceNode *child = part->get_child(
          part->color_space->linearize_color(color));

        DomainPoint lo; lo.dim = ndim;
        DomainPoint hi; hi.dim = ndim;
        for (int32_t dim = 0; dim < ndim; ++dim)
        {
          std::vector<coord_t> &extents = all_extents[dim];
          coord_t c = color[dim];
          lo[dim] = extents[c];
          hi[dim] = extents[c + 1] - 1;
        }
        if (child->set_domain(Domain(lo, hi), true/*broadcast*/))
          delete child;
      }

      // Finally, compute the extents of the root index space and return it
      DomainPoint lo; lo.dim = ndim;
      DomainPoint hi; hi.dim = ndim;
      for (int32_t dim = 0; dim < ndim; ++dim)
        hi[dim] = all_extents[dim].back() - 1;

      return Domain(lo, hi);
    }

    //--------------------------------------------------------------------------
    void IndexTask::finalize_output_regions(bool first_invocation)
    //--------------------------------------------------------------------------
    {
      RegionTreeForest *forest = runtime->forest;

      for (unsigned idx = 0; idx < output_regions.size(); ++idx)
      {
        const OutputOptions &options = output_region_options[idx];
        if (options.valid_requirement())
          continue;
        IndexSpaceNode *parent= forest->get_node(
            output_regions[idx].parent.get_index_space());
#ifdef DEBUG_LEGION
        validate_output_extents(idx, output_regions[idx],
                                output_region_extents[idx]);
#endif
        if (options.global_indexing())
        {
          // For globally indexed output regions, we need to check
          // the alignment between outputs from adjacent point tasks
          // and compute the ranges of subregions via prefix sum.

          IndexPartNode *part = runtime->forest->get_node(
            output_regions[idx].partition.get_index_partition());
          Domain root_domain = compute_global_output_ranges(parent, part,
              output_region_extents[idx], output_region_extents[idx]);

          log_index.debug()
            << "[Task " << get_task_name() << "(UID: " << get_unique_op_id()
            << ")] setting " << root_domain << " to index space " << std::hex
            << parent->handle.get_id();

          if (parent->set_domain(root_domain))
            delete parent;
        }
        // For locally indexed output regions, sizes of subregions are already
        // set when they are fianlized by the point tasks. So we only need to
        // initialize the root index space by taking a union of subspaces.
        else if (parent->set_output_union(output_region_extents[idx]))
          delete parent;
      }
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::initialize_task(InnerContext *ctx,
                                         const IndexTaskLauncher &launcher,
                                         IndexSpace launch_sp,
                                         Provenance *provenance,
                                         bool track /*= true*/,
                             std::vector<OutputRequirement> *outputs /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      initialize_regions(launcher.region_requirements);
      futures = launcher.futures;
      // If the task has any output requirements, we create fresh region and
      // partition names and return them back to the user
      if (outputs != NULL)
        create_output_regions(*outputs, launch_sp);
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_LEGION
        assert(arg_manager == NULL);
#endif
        arg_manager = new AllocManager(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, launcher.global_arg.get_ptr(), arglen);
      }
      // Very important that these freezes occur before we initialize
      // this operation because they can launch creation operations to
      // make the future maps
      point_arguments = 
        launcher.argument_map.impl->freeze(parent_ctx, provenance);
      const size_t num_point_futures = launcher.point_futures.size();
      if (num_point_futures > 0)
      {
        point_futures.resize(num_point_futures);
        for (unsigned idx = 0; idx < num_point_futures; idx++)
          point_futures[idx] = 
            launcher.point_futures[idx].impl->freeze(parent_ctx, provenance);
      }
      concurrent_task = launcher.concurrent;
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      is_index_space = true;
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
      assert(launch_space == NULL);
#endif
      launch_space = runtime->forest->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        launch_space->get_domain(index_domain);
      else
        index_domain = launcher.launch_domain;
      internal_space = launch_space->handle;
      sharding_space = launcher.sharding_space;
      initialize_base_task(ctx, launcher.predicate, task_id, provenance);
      if (outputs != NULL)
      {
        if (launcher.predicate != Predicate::TRUE_PRED)
          REPORT_LEGION_ERROR(ERROR_OUTPUT_REGIONS_IN_PREDICATED_TASK,
              "Output requirements are disallowed for tasks launched with "
              "predicates, but preidcated task launch for task %s (%lld) in "
              "parent task %s (UID %lld) is used with output requirements.",
              get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
        if (get_trace() != NULL)
          REPORT_LEGION_ERROR(ERROR_OUTPUT_REGIONS_IN_TRACE,
              "Output requirements are disallowed for tasks launched inside "
              "traces. Task %s (UID %lld) in parent task %s (UID %lld) has "
              "output requirements in trace %d.", get_task_name(), 
              get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id(), get_trace()->get_trace_id())
      }
      if (!launcher.elide_future_return)
      {
        if (launcher.predicate != Predicate::TRUE_PRED)
          initialize_predicate(launcher.predicate_false_future,
                               launcher.predicate_false_result);
        future_map = 
          create_future_map(ctx, launch_space->handle, launcher.sharding_space);
      }
      else
        elide_future_return = true;
      validate_region_requirements(); 
      if (concurrent_task && parent_ctx->is_concurrent_context())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_CONCURRENT_EXECUTION,
            "Illegal nested concurrent index space task launch %s (UID %lld) "
            "inside task %s (UID %lld) which has a concurrent ancesstor (must "
            "epoch or index task). Nested concurrency is not supported.", 
            get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
            parent_ctx->get_unique_id())
      if (runtime->legion_spy_enabled)
      {
        // Don't log this yet if we're part of a must epoch operation
        if (track)
          LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                    unique_op_id, task_id, get_task_name());
        for (std::vector<PhaseBarrier>::const_iterator it = 
              launcher.wait_barriers.begin(); it !=
              launcher.wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
      }
      return future_map;
    }

    //--------------------------------------------------------------------------
    Future IndexTask::initialize_task(InnerContext *ctx,
                                      const IndexTaskLauncher &launcher,
                                      IndexSpace launch_sp,
                                      Provenance *provenance,
                                      ReductionOpID redop_id, 
                                      bool deterministic,
                                      bool track /*= true*/,
                             std::vector<OutputRequirement> *outputs /*= NULL*/)
    //--------------------------------------------------------------------------
    {
      if (launcher.elide_future_return)
      {
        initialize_task(ctx, launcher, launch_sp, provenance, track, outputs);
        return Future();
      }
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      initialize_regions(launcher.region_requirements);
      futures = launcher.futures;
      // If the task has any output requirements, we create fresh region and
      // partition names and return them back to the user
      if (outputs != NULL)
        create_output_regions(*outputs, launch_sp);
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);
      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
#ifdef DEBUG_LEGION
        assert(arg_manager == NULL);
#endif
        arg_manager = new AllocManager(arglen);
        arg_manager->add_reference();
        args = arg_manager->get_allocation();
        memcpy(args, launcher.global_arg.get_ptr(), arglen);
      }
      // Very important that these freezes occur before we initialize
      // this operation because they can launch creation operations to
      // make the future maps
      point_arguments = 
        launcher.argument_map.impl->freeze(parent_ctx, provenance);
      const size_t num_point_futures = launcher.point_futures.size();
      if (num_point_futures > 0)
      {
        point_futures.resize(num_point_futures);
        for (unsigned idx = 0; idx < num_point_futures; idx++)
          point_futures[idx] = 
            launcher.point_futures[idx].impl->freeze(parent_ctx, provenance);
      }
      concurrent_task = launcher.concurrent;
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
#ifdef DEBUG_LEGION
        assert(mapper_data == NULL);
#endif
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      is_index_space = true;
#ifdef DEBUG_LEGION
      assert(launch_sp.exists());
      assert(launch_space == NULL);
#endif
      launch_space = runtime->forest->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        launch_space->get_domain(index_domain);
      else
        index_domain = launcher.launch_domain;
      internal_space = launch_space->handle;
      sharding_space = launcher.sharding_space;
      redop = redop_id;
      reduction_op = Runtime::get_reduction_op(redop);
      redop_initial_value = launcher.initial_value;
      deterministic_redop = deterministic;
      serdez_redop_fns = Runtime::get_serdez_redop_fns(redop);
      if (!reduction_op->identity)
        REPORT_LEGION_ERROR(ERROR_REDUCTION_OPERATION_INDEX,
                      "Reduction operation %d for index task launch %s "
                      "(ID %lld) is not foldable.",
                      redop, get_task_name(), get_unique_id())
      initialize_base_task(ctx, launcher.predicate, task_id, provenance);
      if (outputs != NULL)
      {
        if (launcher.predicate != Predicate::TRUE_PRED)
          REPORT_LEGION_ERROR(ERROR_OUTPUT_REGIONS_IN_PREDICATED_TASK,
              "Output requirements are disallowed for tasks launched with "
              "predicates, but preidcated task launch for task %s (%lld) in "
              "parent task %s (UID %lld) is used with output requirements.",
              get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id())
        if (get_trace() != NULL)
          REPORT_LEGION_ERROR(ERROR_OUTPUT_REGIONS_IN_TRACE,
              "Output requirements are disallowed for tasks launched inside "
              "traces. Task %s (UID %lld) in parent task %s (UID %lld) has "
              "output requirements in trace %d.", get_task_name(), 
              get_unique_id(), parent_ctx->get_task_name(),
              parent_ctx->get_unique_id(), get_trace()->get_trace_id())
      }
      if (launcher.predicate != Predicate::TRUE_PRED)
        initialize_predicate(launcher.predicate_false_future,
                             launcher.predicate_false_result);
      reduction_future = Future(new FutureImpl(parent_ctx, runtime,
          true/*register*/, runtime->get_available_distributed_id(),
          provenance, this));
      if (serdez_redop_fns == NULL)
        reduction_future.impl->set_future_result_size(
            reduction_op->sizeof_rhs, runtime->address_space);
      validate_region_requirements();
      if (concurrent_task && parent_ctx->is_concurrent_context())
        REPORT_LEGION_ERROR(ERROR_ILLEGAL_CONCURRENT_EXECUTION,
            "Illegal nested concurrent index space task launch %s (UID %lld) "
            "inside task %s (UID %lld) which has a concurrent ancesstor (must "
            "epoch or index task). Nested concurrency is not supported.", 
            get_task_name(), get_unique_id(), parent_ctx->get_task_name(),
            parent_ctx->get_unique_id())
      if (runtime->legion_spy_enabled && track)
      {
        LegionSpy::log_index_task(parent_ctx->get_unique_id(),
                                  unique_op_id, task_id, get_task_name());
        for (std::vector<PhaseBarrier>::const_iterator it = 
              launcher.wait_barriers.begin(); it !=
              launcher.wait_barriers.end(); it++)
        {
          ApEvent e = Runtime::get_previous_phase(it->phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
        LegionSpy::log_future_creation(unique_op_id, 
              reduction_future.impl->did, index_point);
      }
      return reduction_future;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_regions(const std::vector<RegionRequirement> &rs)
    //--------------------------------------------------------------------------
    {
      regions = rs;
      // Rewrite any singular region requirements to projections
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        RegionRequirement &req = regions[idx];
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
        {
          req.handle_type = LEGION_REGION_PROJECTION;
          req.projection = 0; // identity
        }
        // These are some checks for sanity if the user is using the default
        // projection functor from an upper bound region to make sure they
        // know what they are doing
        if (IS_WRITE(req) && (req.projection == 0) &&
            (req.handle_type == LEGION_REGION_PROJECTION))
        {
          if (IS_WRITE_DISCARD(req))
          {
            if (!IS_COLLECTIVE(req))
              REPORT_LEGION_ERROR(ERROR_ALIASED_INTERFERING_REGION,
                  "Parent task %s (UID %lld) issued index space task %s "
                  "(UID %lld) with interfering region requirement %d that "
                  "requested write-discard privileges for all point tasks "
                  "on the same logical region without indicating that they "
                  "should be performed concurrently. If you intend for all "
                  "the point tasks to perform independent writes to the same "
                  "logical region then you must mark the region requirement "
                  "as being a collective write.", parent_ctx->get_task_name(),
                  parent_ctx->get_unique_id(), get_task_name(),
                  get_unique_op_id(), idx)
          }
          else if (runtime->runtime_warnings)
            REPORT_LEGION_WARNING(
                LEGION_WARNING_NON_SCALABLE_IDENTITY_PROJECTION,
                "Parent task %s (UID %lld) issued index space task %s "
                "(UID %lld) with non-scalable projection region requirement %d "
                "that ensures all point tasks will be reading and writing to "
                "the same logical region. This implies there will be no task "
                "parallelism in this index space task launch.",
                parent_ctx->get_task_name(), parent_ctx->get_unique_id(),
                get_task_name(), get_unique_op_id(), idx)
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_predicate(const Future &pred_future,
                                         const UntypedBuffer &pred_arg)
    //--------------------------------------------------------------------------
    {
      if (pred_future.impl != NULL)
        predicate_false_future = pred_future;
      else
      {
        predicate_false_size = pred_arg.get_size();
        if (predicate_false_size > 0)
        {
#ifdef DEBUG_LEGION
          assert(predicate_false_result == NULL);
#endif
          predicate_false_result = 
            legion_malloc(PREDICATE_ALLOC, predicate_false_size);
          memcpy(predicate_false_result, pred_arg.get_ptr(),
                 predicate_false_size);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::prepare_map_must_epoch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch != NULL);
#endif
      set_origin_mapped(true);
      total_points = launch_space->get_volume();
      if (!elide_future_return)
      {
        future_map = must_epoch->get_future_map(); 
        enumerate_futures(index_domain);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      // First compute the parent indexes
      compute_parent_indexes(); 
      // Initialize the privilege paths
      if (!options_selected)
      {
        const bool inline_task = select_task_options(false/*prioritize*/);
        if (inline_task) 
        {
          REPORT_LEGION_WARNING(LEGION_WARNING_MAPPER_REQUESTED_INLINE,
                          "Mapper %s requested to inline task %s "
                          "(UID %lld) but the 'enable_inlining' option was "
                          "not set on the task launcher so the request is "
                          "being ignored", mapper->get_mapper_name(),
                          get_task_name(), get_unique_id());
        }
      }
      if (runtime->legion_spy_enabled)
      { 
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
          TaskOp::log_requirement(unique_op_id, idx, logical_regions[idx]);
        runtime->forest->log_launch_space(launch_space->handle, unique_op_id);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      perform_base_dependence_analysis();
      analyze_region_requirements(launch_space);
    }

    //--------------------------------------------------------------------------
    void IndexTask::create_output_regions(
               std::vector<OutputRequirement> &outputs, IndexSpace launch_space)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      size_t num_tasks = runtime->forest->get_domain_volume(launch_space);
#endif
      Provenance *provenance = get_provenance();
      output_region_options.resize(outputs.size());
      output_region_extents.resize(outputs.size());
      for (unsigned idx = 0; idx < outputs.size(); idx++)
      {
        OutputRequirement &req = outputs[idx];
        output_region_options[idx] = 
          OutputOptions(req.global_indexing, req.valid_requirement);

        IndexSpace color_space = launch_space;
        if (req.projection != 0) {
          color_space = req.color_space;

          if (!color_space.exists())
            REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_PROJECTION,
              "Output region %u of task %s (UID: %lld) requests projection "
              "of ID %u but no color space is specified. "
              "Every output requirement with a non-identity projection must "
              "have a color space set.",
              idx, get_task_name(), get_unique_op_id(), req.projection);

#ifdef DEBUG_LEGION
          IndexSpaceNode* node = runtime->forest->get_node(color_space);
          Domain color_domain;
          node->get_domain(color_domain);
          // No need to wait on the ready event since it is tight

          if (req.global_indexing && !color_domain.dense())
            REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_PROJECTION,
              "The global indexing mode requires the color space of an "
              "output requirement to be dense, but a sparse color space is "
              "assigned to output requirement %u of task %s (UID: %lld).",
              idx, get_task_name(), get_unique_op_id());

          if (color_domain.get_volume() != num_tasks)
            REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_PROJECTION,
              "Output region %u of task %s (UID: %lld) requests projection "
              "but the volume of the color space is different from the total "
              "number of point tasks. "
              "The mapping between the launch domain and the subregions must "
              "be bijective.",
              idx, get_task_name(), get_unique_op_id());
#endif
        }
        int color_dim = color_space.get_dim();

        if (!req.valid_requirement)
        {
          TypeTag type_tag;
          int requested_dim =
            Internal::NT_TemplateHelper::get_dim(req.type_tag);
          if (req.global_indexing)
          {
            if (color_dim != requested_dim)
              REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_DOMAIN,
                "Output region %u of task %s (UID: %lld) is requested to have "
                "%d dimensions, but the color space has %d dimensions. "
                "Dimensionalities of output regions must be the same as the "
                "color space's in global indexing mode.",
                idx, get_task_name(), get_unique_op_id(), requested_dim,
                launch_space.get_dim());

            type_tag = req.type_tag;
          }
          else
          {
            // When local indexing is used for the output region,
            // we create an (N+1)-D index space when the color domain is N-D.

            // Before creating the index space, we make sure that
            // the dimensionality (N+1) does not exceed LEGION_MAX_DIM.
            if (color_dim + requested_dim > LEGION_MAX_DIM)
              REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_DOMAIN,
                "Dimensionality of output region %u of task %s (UID: %lld) "
                "exceeded LEGION_MAX_DIM. You may rebuild your code with a "
                "bigger LEGION_MAX_DIM value or reduce dimensionality of "
                "either the color space or the output region.",
                idx, get_task_name(), get_unique_op_id());

            OutputRegionTagCreator creator(&type_tag, color_dim);
            Internal::NT_TemplateHelper::demux<OutputRegionTagCreator>(
                req.type_tag, &creator);
          }

          // Create a deferred index space
          IndexSpace index_space =
            parent_ctx->create_unbound_index_space(type_tag, provenance);

          // Create a pending partition using the launch domain as the color space
          IndexPartition pid = parent_ctx->create_pending_partition(
              index_space, color_space,
              LEGION_DISJOINT_COMPLETE_KIND, LEGION_AUTO_GENERATE_ID, 
              provenance, true/*trust partitioning*/);

          // Create an output region and a partition
          LogicalRegion region = parent_ctx->create_logical_region(
              index_space, req.field_space, false/*local region*/,
              provenance, true/*output region*/);

          LogicalPartition partition =
            runtime->forest->get_logical_partition(region, pid);

          // Set the region and partition back to the output requirement
          // so the caller can use it for downstream tasks
          req.partition = partition;
          req.parent = region;
          req.handle_type = LEGION_PARTITION_PROJECTION;
          req.flags |= LEGION_CREATED_OUTPUT_REQUIREMENT_FLAG;
        }

        req.privilege = LEGION_WRITE_DISCARD;

        // Store the output requirement in the task
        output_regions.push_back(req);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::perform_base_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(memo_state != MEMO_REQ);
#endif 
      if (runtime->check_privileges)
        perform_privilege_checks();
      // To be correct with the new scheduler we also have to 
      // register mapping dependences on futures
      for (std::vector<Future>::const_iterator it = futures.begin();
            it != futures.end(); it++)
        if (it->impl != NULL)
          it->impl->register_dependence(this);
      if (predicate_false_future.impl != NULL)
        predicate_false_future.impl->register_dependence(this);
      // Register mapping dependences on any future maps also
      if (point_arguments.impl != NULL)
        point_arguments.impl->register_dependence(this);
      for (std::vector<FutureMap>::const_iterator it = 
            point_futures.begin(); it != point_futures.end(); it++)
        it->impl->register_dependence(this);
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(this, 
                  wait_barriers, arrive_barriers, must_epoch);
      version_infos.resize(logical_regions.size());
    }

    //--------------------------------------------------------------------------
    void IndexTask::report_interfering_requirements(unsigned idx1,unsigned idx2)
    //--------------------------------------------------------------------------
    {
      // For now we only issue this warning in debug mode, eventually we'll
      // turn this on only when users request it when we do our debug refactor
      if ((logical_regions[idx1].handle_type == LEGION_SINGULAR_PROJECTION) &&
          (logical_regions[idx2].handle_type == LEGION_SINGULAR_PROJECTION))
        REPORT_LEGION_ERROR(ERROR_ALIASED_REGION_REQUIREMENTS,
                          "Aliased region requirements for index tasks "
                          "are not permitted. Region requirements %d and %d "
                          "of task %s (UID %lld) in parent task %s (UID %lld) "
                          "are interfering.", idx1, idx2, get_task_name(),
                          get_unique_id(), parent_ctx->get_task_name(),
                          parent_ctx->get_unique_id())
      // Need a lock here in case this gets called in parallel by multiple
      // slice tasks returning at the same time
      AutoLock o_lock(op_lock);
      interfering_requirements.insert(std::pair<unsigned,unsigned>(idx1,idx2));
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Do a quick test for empty index space launches
      total_points = launch_space->get_volume();
      if (total_points == 0)
      {
        // Clean up this task execution if there are no points
        complete_mapping();
        complete_execution();
        trigger_children_complete();
        trigger_children_committed();
      }
      else
      {
        // Enumerate the futures in the future map
        if ((redop == 0) && !elide_future_return)
          enumerate_futures(index_domain);
        Operation::trigger_ready();
      }
    }

    //--------------------------------------------------------------------------
    size_t IndexTask::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return launch_space->get_volume();
    }

    //--------------------------------------------------------------------------
    void IndexTask::enumerate_futures(const Domain &domain)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!elide_future_return);
      assert(future_handles == NULL);
#endif
      future_handles = new FutureHandles;
      future_handles->add_reference();
      std::map<DomainPoint,DistributedID> &handles = future_handles->handles;
      for (Domain::DomainPointIterator itr(domain); itr; itr++)
      {
        Future f = future_map.impl->get_future(itr.p, true/*internal only*/);
        handles[itr.p] = f.impl->did;
      }
    }

    //--------------------------------------------------------------------------
    RtEvent IndexTask::verify_concurrent_execution(const DomainPoint &point,
                                                   Processor target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(concurrent_task);
#endif
      AutoLock o_lock(op_lock);
      if (concurrent_processors.empty())
      {
#ifdef DEBUG_LEGION
        assert(!concurrent_verified.exists());
#endif
        concurrent_verified = Runtime::create_rt_user_event();
      }
#ifdef DEBUG_LEGION
      assert(concurrent_processors.find(point) == 
              concurrent_processors.end());
      assert(concurrent_processors.size() < total_points);
#endif
      concurrent_processors[point] = target;
      if (concurrent_processors.size() == total_points)
      {
        std::map<Processor,DomainPoint> inverted;
        for (std::map<DomainPoint,Processor>::const_iterator it =
              concurrent_processors.begin(); it != 
              concurrent_processors.end(); it++)
        {
          std::map<Processor,DomainPoint>::const_iterator finder = 
            inverted.find(it->second);
          if (finder != inverted.end())
          {
            if (mapper == NULL)
              mapper = runtime->find_mapper(current_proc, map_id);
            // TODO: update this error message to name the bad points
            REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                "Mapper %s performed illegal mapping of concurrent index "
                "space task %s (UID %lld) by mapping multiple points to "
                "the same processor " IDFMT ". All point tasks must be "
                "mapped to different processors for concurrent execution "
                "of index space tasks.", mapper->get_mapper_name(),
                get_task_name(), get_unique_id(), it->second.id)
          }
          inverted[it->second] = it->first;
        }
        Runtime::trigger_event(concurrent_verified);
      }
      return concurrent_verified;
    }

    //--------------------------------------------------------------------------
    void IndexTask::concurrent_allreduce(SliceTask *slice,
        AddressSpaceID slice_space, size_t points, uint64_t lamport_clock,
        VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      {
        AutoLock o_lock(op_lock);
        if (concurrent_lamport_clock < lamport_clock)
          concurrent_lamport_clock = lamport_clock;
        if (poisoned)
          concurrent_poisoned = true;
        concurrent_slices.push_back(std::make_pair(slice, slice_space));
        if (concurrent_points == 0)
          concurrent_variant = vid;
        else if (concurrent_variant != vid)
          concurrent_variant = std::min(concurrent_variant, vid);
        concurrent_points += points;
        done = (concurrent_points == total_points);
      }
      if (done)
      {
        if (concurrent_variant > 0)
        {
          VariantImpl *variant = 
            runtime->find_variant_impl(task_id, concurrent_variant);
          if (variant->needs_barrier())
            concurrent_task_barrier =
              RtBarrier(Realm::Barrier::create_barrier(total_points));
        }
        // Swap this vector onto the stack in case the slice task gets deleted
        // out from under us while we are finalizing things
        std::vector<std::pair<SliceTask*,AddressSpaceID> > local_copy;
        local_copy.swap(concurrent_slices);
        for (std::vector<std::pair<SliceTask*,AddressSpaceID> >::const_iterator
              it = local_copy.begin(); it != local_copy.end(); it++)
        {
          if (it->second != runtime->address_space)
          {
            Serializer rez;
            {
              RezCheck z(rez);
              rez.serialize(it->first);
              rez.serialize(concurrent_task_barrier);
              rez.serialize(concurrent_lamport_clock);
              rez.serialize(concurrent_variant);
              rez.serialize(concurrent_poisoned);
            }
            runtime->send_slice_concurrent_allreduce_response(it->second, rez);
          }
          else
            it->first->finish_concurrent_allreduce(
                concurrent_lamport_clock, concurrent_poisoned, 
                concurrent_variant, concurrent_task_barrier);
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      RtEvent execution_condition;
      // Fill in the index task map with the default future value
      if (!elide_future_return)
      {
        if (redop == 0)
        {
          // Only need to do this if the internal domain exists, it
          // might not in a control replication context
          if (internal_space.exists())
          {
            // Get the domain that we will have to iterate over
            Domain local_domain;
            runtime->forest->find_domain(internal_space, local_domain);
            // Handling the future map case
            if (predicate_false_future.impl != NULL)
            {
              for (Domain::DomainPointIterator itr(local_domain);
                    itr; itr++)
              {
                Future f = future_map.impl->get_future(itr.p,
                                            true/*internal*/);
                f.impl->set_result(predicate_false_future.impl, this);
              }
            }
            else
            {
              for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
              {
                Future f = future_map.impl->get_future(itr.p, true/*internal*/);
                if (predicate_false_size > 0)
                  f.impl->set_local(predicate_false_result,
                                    predicate_false_size, false/*own*/);
                else
                  f.impl->set_result(ApEvent::NO_AP_EVENT, NULL);
              }
            }
          }
        }
        else
        {
          // Handling a reduction case
          if (redop_initial_value.impl == NULL)
            reduction_future.impl->set_local(&reduction_op->identity,
                                             reduction_op->sizeof_rhs);
          else
            reduction_future.impl->set_result(redop_initial_value.impl, this);
        }
      }
      // Then clean up this task execution
      complete_mapping();
      complete_execution(execution_condition);
      trigger_children_complete();
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::premap_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_PREMAP_TASK_CALL);
      // We only need to premap the task if it has a reduction down
      // to individual futures so that we can map the futures
      if (redop == 0)
        return;
      // Call premap task here to see if there are any future destinations
      Mapper::PremapTaskInput input;
      Mapper::PremapTaskOutput output;
      // Initialize this to not have a new target processor
      output.new_target_proc = Processor::NO_PROC;
      // Now invoke the mapper call
      if (mapper == NULL)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_premap_task(this, input, output);
      // See if we need to update the new target processor
      if (output.new_target_proc.exists())
        this->target_proc = output.new_target_proc;
      create_future_instances(output.reduction_futures);
      // If we're recording this trace then we need to remember this
      if (is_recording())
      {
#ifdef DEBUG_LEGION
        assert((tpl != NULL) && tpl->is_recording());
#endif
        tpl->record_premap_output(this, output, map_applied_conditions);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::create_future_instances(std::vector<Memory> &target_mems)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(reduction_instances.empty());
#endif
      if (!target_mems.empty())
      {
        if (target_mems.size() > 1)
        {
          std::set<Memory> unique_mems;
          for (std::vector<Memory>::iterator it =
                target_mems.begin(); it != target_mems.end(); /*nothing*/)
          {
            if (!it->exists())
              REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                  "Invalid mapper output. Mapper %s requested index task "
                  "reduction future be mapped to a NO_MEMORY for task %s "
                  "(UID %lld) which is illegal. All requests for mapping "
                  "output futures must be mapped to actual memories.",
                  mapper->get_mapper_name(), get_task_name(), unique_op_id)
            if (unique_mems.find(*it) == unique_mems.end())
            {
              unique_mems.insert(*it);
              it++;
            }
            else
              it = target_mems.erase(it);
          }
        }
        else if (!(target_mems.begin()->exists()))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                "Invalid mapper output. Mapper %s requested index task "
                "reduction future be mapped to a NO_MEMORY for task %s "
                "(UID %lld) which is illegal. All requests for mapping "
                "output futures must be mapped to actual memories.",
                mapper->get_mapper_name(), get_task_name(), unique_op_id)
      }
      else
        target_mems.push_back(runtime->runtime_system_memory);
      // If we've got a serdez redop function then we don't know how big
      // the output is going to be until later, otherwise we know the
      // output size from the reduction operator
      if (serdez_redop_fns == NULL) 
      {
        reduction_instances.reserve(target_mems.size());
        int runtime_visible_index = -1;
        for (std::vector<Memory>::const_iterator it =
              target_mems.begin(); it != target_mems.end(); it++)
        {
          if ((runtime_visible_index < 0) &&
              FutureInstance::check_meta_visible(*it))
            runtime_visible_index = reduction_instances.size();
          MemoryManager *manager = runtime->find_memory_manager(*it);
          reduction_instances.push_back(
              manager->create_future_instance(this, unique_op_id,
                reduction_op->sizeof_rhs, false/*eager*/));
        }
        // This is an important optimization: if we're doing a small
        // reduction value we always want the reduction instance to
        // be somewhere meta visible for performance reasons, so we
        // make a meta-visible instance if we don't have one
        if ((runtime_visible_index < 0) &&
            (reduction_op->sizeof_rhs <= LEGION_MAX_RETURN_SIZE))
        {
          runtime_visible_index = reduction_instances.size();
          MemoryManager *manager = 
            runtime->find_memory_manager(runtime->runtime_system_memory);
          reduction_instances.push_back(
              manager->create_future_instance(this, unique_op_id,
                reduction_op->sizeof_rhs, false/*eager*/));
        }
        if (runtime_visible_index > 0)
          std::swap(reduction_instances.front(), 
              reduction_instances[runtime_visible_index]);
#ifdef DEBUG_LEGION
        assert(reduction_instance == NULL);
#endif
        reduction_instance = reduction_instances.front();
        // Need to initialize this with the reduction value
        if ((redop_initial_value.impl != NULL) &&
            (parent_ctx->get_task()->get_shard_id() == 0))
          reduction_instance_precondition =
            redop_initial_value.impl->copy_to(reduction_instance, this);
        else
          reduction_instance_precondition =
            reduction_instance.load()->initialize(reduction_op, this);
      }
      else
      {
        if ((redop_initial_value.impl != NULL) &&
            (parent_ctx->get_task()->get_shard_id() == 0))
        {
          const RtEvent ready = 
            redop_initial_value.impl->request_runtime_instance(this, false);
          if (ready.exists() && !ready.has_triggered())
            ready.wait();
          const void *value = redop_initial_value.impl->find_runtime_buffer(
              parent_ctx, serdez_redop_state_size); 
          serdez_redop_state = malloc(serdez_redop_state_size);
          memcpy(serdez_redop_state, value, serdez_redop_state_size);
        }
        serdez_redop_targets.swap(target_mems);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_DISTRIBUTE_CALL);
      if (is_origin_mapped())
      {
        // This will only get called if we had slices that couldn't map, but
        // they have now all mapped
#ifdef DEBUG_LEGION
        assert(slices.empty());
#endif
        // We're never actually run
        return false;
      }
      else
      {
        if (!is_sliced() && target_proc.exists() && 
            (target_proc != current_proc))
        {
          // Make a slice copy and send it away
          SliceTask *clone = clone_as_slice_task(internal_space, target_proc,
                                                 true/*needs slice*/,
                                                 stealable);
          runtime->send_task(clone);
          return false; // We have now been sent away
        }
        else
          return true; // Still local so we can be sliced
      }
    }

    //--------------------------------------------------------------------------
    RtEvent IndexTask::perform_mapping(MustEpochOp *owner/*=NULL*/,
                                       const DeferMappingArgs *args/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_PERFORM_MAPPING_CALL);
      // This will only get called if we had slices that failed to origin map 
#ifdef DEBUG_LEGION
      assert(!slices.empty());
      // Should never get duplicate invocations here
      assert(args == NULL);
#endif
      for (std::list<SliceTask*>::iterator it = slices.begin();
            it != slices.end(); /*nothing*/)
      {
        (*it)->trigger_mapping();
        it = slices.erase(it);
      }
      return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    void IndexTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    bool IndexTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      // Index space tasks are never stealable, they must first be
      // split into slices which can then be stolen.  Note that slicing
      // always happens after premapping so we know stealing is safe.
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      // This should only ever be called if we had slices which failed to map
#ifdef DEBUG_LEGION
      assert(is_sliced());
      assert(!slices.empty());
#endif
      trigger_slices();
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind IndexTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INDEX_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_COMPLETE_CALL);
      // Trigger all the futures or set the reduction future result
      // and then trigger it
      if (redop != 0)
      {
        // Set the future if we actually ran the task or we speculated
        if (predication_state != PREDICATED_FALSE_STATE)
        {
#ifdef DEBUG_LEGION
          assert(!reduction_instances.empty());
          assert(reduction_instance == reduction_instances.front());
          assert(reduction_fold_effects.empty());
#endif
          // Now do the copy out from the reduction_instance to any other
          // target futures that we have, we'll do this with a broadcast tree
          if (reduction_instances.size() > 1)
          {
            std::vector<ApEvent> reduction_instances_ready(
                reduction_instances.size(), reduction_instance_precondition);
            // Do the copy from 0 to 1 first
            reduction_instances_ready[1] = reduction_instances[1]->copy_from(
                reduction_instance, this, reduction_instances_ready[0]);
            for (unsigned idx = 1; idx < reduction_instances.size(); idx++)
            {
              if (reduction_instances.size() <= (2*idx))
                break;
              reduction_instances_ready[2*idx] =
                reduction_instances[2*idx]->copy_from(reduction_instances[idx],
                  this, reduction_instances_ready[idx]);
              if (reduction_instances.size() <= (2*idx+1))
                break;
              reduction_instances_ready[2*idx+1] =
               reduction_instances[2*idx+1]->copy_from(reduction_instances[idx],
                 this, reduction_instances_ready[idx]);
            }
            record_completion_effects(reduction_instances_ready);
          }
          else
            record_completion_effect(reduction_instance_precondition);
          reduction_future.impl->set_results(get_completion_event(),
              reduction_instances, reduction_metadata, reduction_metasize);
          // Clear this since we no longer own the buffer
          reduction_metadata = NULL;
          reduction_instances.clear();
        }
      }
#ifdef LEGION_SPY
      LegionSpy::log_operation_events(unique_op_id, ApEvent::NO_AP_EVENT,
                                      get_completion_event());
#endif
      if (must_epoch != NULL)
      {
        RtEvent precondition;
        if (!complete_preconditions.empty())
          precondition = Runtime::merge_events(complete_preconditions);
        must_epoch->notify_subop_complete(this, precondition);
        complete_operation();
      } 
      else
      {
        if (!complete_preconditions.empty())
          complete_operation(Runtime::merge_events(complete_preconditions));
        else
          complete_operation();
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_COMMIT_CALL); 
      if (profiling_reported.exists())
      {
        if (outstanding_profiling_requests.load() > 0)
        {
#ifdef DEBUG_LEGION
          assert(mapped_event.has_triggered());
#endif
          std::vector<IndexProfilingInfo> to_perform;
          {
            AutoLock o_lock(op_lock);
            to_perform.swap(profiling_info);
          }
          if (!to_perform.empty())
          {
            for (unsigned idx = 0; idx < to_perform.size(); idx++)
            {
              IndexProfilingInfo &info = to_perform[idx];
              const Realm::ProfilingResponse resp(info.buffer,info.buffer_size);
              info.total_reports = outstanding_profiling_requests.load();
              info.profiling_responses.attach_realm_profiling_response(resp);
              mapper->invoke_task_report_profiling(this, info);
              free(info.buffer);
            }
            const int count = to_perform.size() +
                outstanding_profiling_reported.fetch_add(to_perform.size());
#ifdef DEBUG_LEGION
            assert(count <= outstanding_profiling_requests.load());
#endif
            if (count == outstanding_profiling_requests.load())
              Runtime::trigger_event(profiling_reported);
          }
        }
        else
        {
          // We're not expecting any profiling callbacks so we need to
          // do one ourself to inform the mapper that there won't be any
          Mapping::Mapper::TaskProfilingInfo info;
          info.total_reports = 0;
          info.task_response = true;
          info.region_requirement_index = 0;
          info.fill_response = false; // make valgrind happy
          mapper->invoke_task_report_profiling(this, info);    
          Runtime::trigger_event(profiling_reported);
        }
        commit_preconditions.insert(profiling_reported);
      }
      // If we have an origin-mapped slices then we need to check to see
      // if we're waiting on any profiling reports from them
      if (!origin_mapped_slices.empty())
      {
        for (std::set<SliceTask*>::const_iterator it = 
              origin_mapped_slices.begin(); it != 
              origin_mapped_slices.end(); it++)
          (*it)->find_commit_preconditions(commit_preconditions);
      }
      if (must_epoch != NULL)
      {
        RtEvent commit_precondition;
        if (!commit_preconditions.empty())
          commit_precondition = Runtime::merge_events(commit_preconditions);
        must_epoch->notify_subop_commit(this, commit_precondition);
        commit_operation(true/*deactivate*/, commit_precondition);
      }
      else
      {
        // Mark that this operation is now committed
        if (!commit_preconditions.empty())
          commit_operation(true/*deactivate*/, 
              Runtime::merge_events(commit_preconditions));
        else
          commit_operation(true/*deactivate*/);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::pack_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    bool IndexTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexTask::perform_inlining(VariantImpl *variant,
                                  const std::deque<InstanceSet> &parent_regions)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_PERFORM_INLINING_CALL);
      total_points = launch_space->get_volume();
      if ((redop == 0) && !elide_future_return)
        enumerate_futures(index_domain);
      SliceTask *slice = clone_as_slice_task(launch_space->handle,
                current_proc, false/*recurse*/, false/*stealable*/);
      slice->enumerate_points(true/*inlining*/);
      slice->perform_inlining(variant, parent_regions);
    }

    //--------------------------------------------------------------------------
    SliceTask* IndexTask::clone_as_slice_task(IndexSpace is, Processor p,
                                              bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_CLONE_AS_SLICE_CALL);
      SliceTask *result = runtime->get_available_slice_task(); 
      result->initialize_base_task(parent_ctx, Predicate::TRUE_PRED,
                                   this->task_id, get_provenance());
      result->clone_multi_from(this, is, p, recurse, stealable);
      result->index_owner = this;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_index_slice(get_unique_id(), 
                                   result->get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(get_unique_op_id(),
                                                result->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexTask::reduce_future(const DomainPoint &point,
                                  FutureInstance *inst, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_HANDLE_FUTURE);
#ifdef DEBUG_LEGION
      assert(reduction_op != NULL);
#endif
      // If we're doing a deterministic reduction then we need to 
      // buffer up these future values until we get all of them so
      // that we can fold them in a deterministic way
      if (deterministic_redop)
      {
        // Store it in our temporary futures for later
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(temporary_futures.find(point) == temporary_futures.end());
#endif
        temporary_futures[point] = std::make_pair(inst, effects);
      }
      else
      {
        if (!fold_reduction_future(inst, effects))
        {
          // save it to delete later
          AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
          assert(temporary_futures.find(point) == temporary_futures.end());
#endif
          temporary_futures[point] = std::make_pair(inst, effects);
        }
        else
          delete inst;
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::pack_profiling_requests(Serializer &rez,
                                            std::set<RtEvent> &applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(copy_fill_priority);
      rez.serialize<size_t>(copy_profiling_requests.size());
      if (!copy_profiling_requests.empty())
      {
        for (unsigned idx = 0; idx < copy_profiling_requests.size(); idx++)
          rez.serialize(copy_profiling_requests[idx]);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Send a message to the owner with an update for the extra counts
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        rez.serialize<RtEvent>(done_event);
        applied.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    int IndexTask::add_copy_profiling_request(const PhysicalTraceInfo &info,
                Realm::ProfilingRequestSet &requests, bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any copy profiling requests
      if (copy_profiling_requests.empty())
        return copy_fill_priority;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest &request = requests.add_request(
        runtime->find_utility_group(), LG_LEGION_PROFILING_ID, 
        &response, sizeof(response));
      for (std::vector<ProfilingMeasurementID>::const_iterator it = 
            copy_profiling_requests.begin(); it != 
            copy_profiling_requests.end(); it++)
        request.add_measurement((Realm::ProfilingMeasurementID)(*it));
      handle_profiling_update(count);
      return copy_fill_priority;
    }

    //--------------------------------------------------------------------------
    void IndexTask::handle_profiling_response(
                                       const ProfilingResponseBase *base,
                                       const Realm::ProfilingResponse &response,
                                       const void *orig, size_t orig_length)
    //--------------------------------------------------------------------------
    {
      const OpProfilingResponse *task_prof = 
            static_cast<const OpProfilingResponse*>(base);
      // Check to see if we are done mapping, if not then we need to defer
      // this until we are done mapping so we know how many
      if (!mapped_event.has_triggered())
      {
        // Take the lock and see if we lost the race
        AutoLock o_lock(op_lock);
        if (!mapped_event.has_triggered())
        {
          // Save this profiling response for later until we know the
          // full count of profiling responses
          profiling_info.resize(profiling_info.size() + 1);
          IndexProfilingInfo &info = profiling_info.back();
          info.task_response = task_prof->task; 
          info.region_requirement_index = task_prof->src;
          info.fill_response = task_prof->fill;
          info.buffer_size = orig_length;
          info.buffer = malloc(orig_length);
          memcpy(info.buffer, orig, orig_length);
          return;
        }
      }
      // If we get here then we can handle the response now
      Mapping::Mapper::TaskProfilingInfo info;
      info.profiling_responses.attach_realm_profiling_response(response);
      info.task_response = task_prof->task; 
      info.region_requirement_index = task_prof->src;
      info.total_reports = outstanding_profiling_requests.load();
      info.fill_response = task_prof->fill;
      mapper->invoke_task_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
#ifdef DEBUG_LEGION
      assert(count <= outstanding_profiling_requests.load());
#endif
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    } 

    //--------------------------------------------------------------------------
    void IndexTask::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(count > 0);
      assert(!mapped_event.has_triggered());
#endif
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void IndexTask::register_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::create_future_map(TaskContext *ctx,
                             IndexSpace launch_space, IndexSpace sharding_space) 
    //--------------------------------------------------------------------------
    {
      FutureMapImpl *result = new FutureMapImpl(ctx, this, this->launch_space,
          runtime, runtime->get_available_distributed_id(), get_provenance());
      future_map_coordinate = result->future_coordinate;
      return FutureMap(result);
    }

    //--------------------------------------------------------------------------
    RtEvent IndexTask::find_intra_space_dependence(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      // Check to see if we already have it
      std::map<DomainPoint,RtEvent>::const_iterator finder = 
        intra_space_dependences.find(point);
      if (finder != intra_space_dependences.end())
        return finder->second;
      // Otherwise make a temporary one and record it for now
      const RtUserEvent pending_event = Runtime::create_rt_user_event();
      intra_space_dependences[point] = pending_event;
      pending_intra_space_dependences[point] = pending_event;
      return pending_event;
    }
    
    //--------------------------------------------------------------------------
    void IndexTask::record_intra_space_dependence(const DomainPoint &point,
                                                  const DomainPoint &next,
                                                  RtEvent point_mapped)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::map<DomainPoint,RtEvent>::iterator finder = 
        intra_space_dependences.find(point);
      if (finder != intra_space_dependences.end())
      {
        if (finder->second != point_mapped)
        {
          std::map<DomainPoint,RtUserEvent>::iterator pending_finder = 
            pending_intra_space_dependences.find(point);
#ifdef DEBUG_LEGION
          assert(pending_finder != pending_intra_space_dependences.end());
#endif
          Runtime::trigger_event(pending_finder->second, point_mapped);
          pending_intra_space_dependences.erase(pending_finder);
          finder->second = point_mapped;
        }
      }
      else
        intra_space_dependences[point] = point_mapped;
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_origin_mapped_slice(SliceTask *local_slice)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      origin_mapped_slices.insert(local_slice);
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_mapped(unsigned points,
                              RtEvent applied_condition, ApEvent slice_complete)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_RETURN_SLICE_MAPPED_CALL);
      bool need_trigger = false;
      bool trigger_children_completed = false;
      bool trigger_children_commit = false;
      record_completion_effect(slice_complete);
      {
        AutoLock o_lock(op_lock);
        mapped_points += points;
        if (applied_condition.exists())
          map_applied_conditions.insert(applied_condition);
        // Already know that mapped points is the same as total points
        if (mapped_points == total_points)
        {
          // Don't complete this yet if we have redop serdez fns because
          // we still need to map the output future instance before we
          // can consider ourselves mapped and we can't do that until we
          // get the final future value
          if (serdez_redop_fns == NULL)
            need_trigger = true;
          if ((complete_points == total_points) &&
              !children_complete_invoked)
          {
            trigger_children_completed = true;
            children_complete_invoked = true;
          }
          if ((committed_points == total_points) &&
              !children_commit_invoked)
          {
            trigger_children_commit = true;
            children_commit_invoked = true;
          }
        }
      }
      if (need_trigger)
      {
        // Get the mapped precondition note we can now access this
        // without holding the lock because we know we've seen
        // all the responses so no one else will be mutating it.
        if (!map_applied_conditions.empty())
        {
          RtEvent map_condition = Runtime::merge_events(map_applied_conditions);
          complete_mapping(map_condition);
        }
        else
          complete_mapping();
      }
      if (trigger_children_completed)
        trigger_children_complete();
      if (trigger_children_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_complete(unsigned points, RtEvent slice_done,
                               void *metadata/*= NULL*/, size_t metasize/*= 0*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_RETURN_SLICE_COMPLETE_CALL);
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (slice_done.exists())
          complete_preconditions.insert(slice_done);
        complete_points += points;
#ifdef DEBUG_LEGION
        assert(complete_points <= total_points);
#endif
        if ((complete_points == total_points) && !children_complete_invoked)
        {
          need_trigger = true;
          children_complete_invoked = true;
        }
        if (metadata != NULL)
        {
#ifdef DEBUG_LEGION
          assert(redop > 0);
#endif
          if (reduction_metadata == NULL)
          {
            reduction_metadata = metadata;
            reduction_metasize = metasize;
            metadata = NULL; // mark that we grabbed it
          }
        }
      }
      if (need_trigger)
      {
        // If we are reducing to a single value we need to finish that now
        if (redop > 0)
        {
#ifdef DEBUG_LEGION
          assert((serdez_redop_fns != NULL) || !reduction_instances.empty());
          assert((serdez_redop_fns != NULL) ||
                  (reduction_instance == reduction_instances.front()));
#endif
          // First finish applying any deterministic reductions
          if (deterministic_redop)
          {
            // Fold any temporary future for deterministic reduction
            for (std::map<DomainPoint,
                  std::pair<FutureInstance*,ApEvent> >::iterator it =
                  temporary_futures.begin(); it != 
                  temporary_futures.end(); /*nothing*/)
            {
              if (fold_reduction_future(it->second.first, it->second.second))
              {
                delete it->second.first;
                std::map<DomainPoint,
                  std::pair<FutureInstance*,ApEvent> >::iterator
                    to_delete = it++;
                temporary_futures.erase(to_delete);
              }
              else
                it++;
            }
          }
          else if (serdez_redop_fns == NULL)
          {
            // Merge any reduction fold events back into the 
            // reduction_instance_precondition to know when the
            // reduction instance is safe to use
            // Note all these events dominate the reduction fold precondition
            // so there is no need to include and we can just overwrite it
            if (!reduction_fold_effects.empty())
            {
              reduction_instance_precondition =
                Runtime::merge_events(NULL, reduction_fold_effects);
              reduction_fold_effects.clear();
            }
          }
#ifdef DEBUG_LEGION
          assert(reduction_fold_effects.empty());
#endif
          // Finish the index task reduction
          finish_index_task_reduction();
        }
        complete_execution();
        trigger_children_complete();
      }
      // If we didn't grab ownership then free this now
      if (metadata != NULL)
        free(metadata);
    }

    //--------------------------------------------------------------------------
    void IndexTask::finish_index_task_reduction(void)
    //--------------------------------------------------------------------------
    {
      // If we have serdez redop fns, we now know how big the output
      // is so we can make our target instances and complete the mapping
      if (serdez_redop_fns != NULL)
      {
#ifdef DEBUG_LEGION
        assert(reduction_instances.empty());
        assert(!serdez_redop_targets.empty());
#endif
        reduction_instances.reserve(serdez_redop_targets.size());
        int runtime_visible_index = -1;
        for (std::vector<Memory>::const_iterator it =
              serdez_redop_targets.begin(); it !=
              serdez_redop_targets.end(); it++)
        {
          if ((runtime_visible_index == -1) && 
              ((*it) == runtime->runtime_system_memory))
          {
            runtime_visible_index = reduction_instances.size();
            reduction_instances.push_back(
                FutureInstance::create_local(serdez_redop_state, 
                  serdez_redop_state_size, false/*own*/));
          }
          else
          {
            MemoryManager *manager = runtime->find_memory_manager(*it);
            reduction_instances.push_back(
                manager->create_future_instance(this, unique_op_id,
                  serdez_redop_state_size, false/*eager*/));
          }
        }
        if (runtime_visible_index < 0)
        {
          runtime_visible_index = reduction_instances.size();
          reduction_instances.push_back(
                FutureInstance::create_local(serdez_redop_state, 
                  serdez_redop_state_size, false/*own*/));
        }
        // Make sure the instance with the data is at the front
        if (runtime_visible_index > 0)
          std::swap(reduction_instances.front(),
              reduction_instances[runtime_visible_index]);
        reduction_instance = reduction_instances.front();
        // Get the mapped precondition note we can now access this
        // without holding the lock because we know we've seen
        // all the responses so no one else will be mutating it.
        if (!map_applied_conditions.empty())
        {
          const RtEvent map_condition = 
            Runtime::merge_events(map_applied_conditions);
          complete_mapping(map_condition);
        }
        else
          complete_mapping();
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_commit(unsigned points, 
                                        RtEvent commit_precondition)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, INDEX_RETURN_SLICE_COMMIT_CALL);
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (commit_precondition.exists())
          commit_preconditions.insert(commit_precondition);
        committed_points += points;
#ifdef DEBUG_LEGION
        assert(committed_points <= total_points);
#endif
        if ((committed_points == total_points) && !children_commit_invoked)
        {
          need_trigger = true;
          children_commit_invoked = true;
        }
      }
      if (need_trigger)
        trigger_children_committed();
    } 

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_mapped(Deserializer &derez, 
                                        AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      RtEvent applied_condition;
      derez.deserialize(applied_condition);
      ApEvent restrict_postcondition;
      derez.deserialize(restrict_postcondition);
#ifdef DEBUG_LEGION
      if (!is_origin_mapped())
      {
        std::map<DomainPoint,std::vector<LogicalRegion> > local_requirements;
        for (unsigned idx = 0; idx < points; idx++)
        {
          DomainPoint point;
          derez.deserialize(point);
          std::vector<LogicalRegion> &reqs = local_requirements[point];
          reqs.resize(regions.size());
          for (unsigned idx2 = 0; idx2 < regions.size(); idx2++)
            derez.deserialize(reqs[idx2]);
        }
        check_point_requirements(local_requirements);
      }
#endif
      return_slice_mapped(points, applied_condition, restrict_postcondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      RtEvent complete_precondition;
      derez.deserialize(complete_precondition);
      const RtEvent resources_returned = (must_epoch == NULL) ?
        ResourceTracker::unpack_resources_return(derez, parent_ctx) :
        ResourceTracker::unpack_resources_return(derez, must_epoch);
      ApEvent completion_effect;
      derez.deserialize(completion_effect);
      record_completion_effect(completion_effect);
      if (redop > 0)
      {
#ifdef DEBUG_LEGION
        assert(reduction_op != NULL);
#endif
        if (deterministic_redop)
        {
          size_t num_futures;
          derez.deserialize(num_futures);
          for (unsigned idx = 0; idx < num_futures; idx++)
          {
            DomainPoint point;
            derez.deserialize(point);
            FutureInstance *instance = FutureInstance::unpack_instance(derez);
            ApEvent effects;
            if (!instance->is_meta_visible)
              derez.deserialize(effects);
            reduce_future(point, instance, effects);
          }
        }
        else
        {
          if (serdez_redop_fns != NULL)
          {
            size_t reduc_size;
            derez.deserialize(reduc_size);
            if (reduc_size > 0)
            {
              const void *reduc_ptr = derez.get_current_pointer();
              FutureInstance instance(reduc_ptr, reduc_size, false/*eager*/,
                  true/*external*/, false/*own allocation*/);
              fold_reduction_future(&instance, ApEvent::NO_AP_EVENT);
              // Advance the pointer on the deserializer
              derez.advance_pointer(reduc_size);
            }
          }
          else
          {
            DomainPoint point;
            derez.deserialize(point);
            if (point.get_dim() > 0)
            {
              FutureInstance *instance = FutureInstance::unpack_instance(derez);
              ApEvent effects;
              if (!instance->is_meta_visible)
                derez.deserialize(effects);
              reduce_future(point, instance, effects);
            }
          }
        }
        size_t metasize;
        derez.deserialize(metasize);
        if (metasize > 0)
        {
          AutoLock o_lock(op_lock);
          if (reduction_metadata == NULL)
          {
            reduction_metadata = malloc(metasize);
            memcpy(reduction_metadata, derez.get_current_pointer(), metasize);
            reduction_metasize = metasize;
          }
          derez.advance_pointer(metasize);
        }
      }

      if (resources_returned.exists())
      {
        if (complete_precondition.exists())
          return_slice_complete(points, Runtime::merge_events(
                complete_precondition, resources_returned));
        else
          return_slice_complete(points, resources_returned);
      }
      else
        return_slice_complete(points, complete_precondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      RtEvent commit_precondition;
      derez.deserialize(commit_precondition);
      return_slice_commit(points, commit_precondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_collective_versioning_rendezvous(
        Deserializer &derez, unsigned index, size_t total_points)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      LegionMap<LogicalRegion,RegionVersioning> to_perform;
      {
        size_t num_regions;
        derez.deserialize(num_regions);
        AutoLock o_lock(op_lock);
        std::map<unsigned,PendingVersioning>::iterator finder =
          pending_versioning.find(index);
        if (finder == pending_versioning.end())
        {
          finder = pending_versioning.insert(
              std::make_pair(index, PendingVersioning())).first;
          finder->second.remaining_arrivals = this->get_collective_points();
        }
        for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
        {
          LogicalRegion region;
          derez.deserialize(region);
          RtUserEvent ready_event;
          derez.deserialize(ready_event);
          LegionMap<LogicalRegion,RegionVersioning>::iterator region_finder =
            finder->second.region_versioning.find(region);
          if (region_finder == finder->second.region_versioning.end())
          {
            region_finder = finder->second.region_versioning.emplace(
                std::make_pair(region,RegionVersioning())).first;
            region_finder->second.ready_event = ready_event;
          }
          else
            Runtime::trigger_event(ready_event,
                region_finder->second.ready_event);
          size_t num_trackers;
          derez.deserialize(num_trackers);
          for (unsigned idx2 = 0; idx2 < num_trackers; idx2++)
          {
            std::pair<AddressSpaceID,EqSetTracker*> key;
            derez.deserialize(key.first);
            derez.deserialize(key.second);
#ifdef DEBUG_LEGION
            assert(region_finder->second.trackers.find(key) ==
                    region_finder->second.trackers.end());
#endif
            derez.deserialize(region_finder->second.trackers[key]);
          }
        }
#ifdef DEBUG_LEGION
        assert(finder->second.remaining_arrivals >= total_points);
#endif
        finder->second.remaining_arrivals -= total_points;
        if (finder->second.remaining_arrivals == 0)
        {
          done = true;
          to_perform.swap(finder->second.region_versioning);
          pending_versioning.erase(finder);
        }
        if (num_regions == 0)
        {
          RtUserEvent done_event;
          derez.deserialize(done_event);
          Runtime::trigger_event(done_event);
        }
      }
      if (done)
      {
        const unsigned parent_req_index = find_parent_index(index);
        finalize_collective_versioning_analysis(index, parent_req_index,
                                                to_perform);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_replaying());
      assert(current_proc.exists());
#endif
#ifdef LEGION_SPY
      LegionSpy::log_replay_operation(unique_op_id);
#endif
      tpl->register_operation(this);
      // If we're going to be doing an output reduction do that now
      if (redop > 0)
      {
        std::vector<Memory> reduction_futures;
        tpl->get_premap_output(this, reduction_futures);
        create_future_instances(reduction_futures); 
      }
      else if (!elide_future_return)
      {
        Domain internal_domain;
        runtime->forest->find_domain(internal_space, internal_domain);
        enumerate_futures(internal_domain);
      }
      // Mark that this is origin mapped effectively in case we
      // have any remote tasks, do this before we clone it
      map_origin = true;
      SliceTask *new_slice = this->clone_as_slice_task(internal_space,
                                                       current_proc,
                                                       false, false);
      // Count how many total points we need for this index space task
      total_points = new_slice->enumerate_points(false/*inline*/);
      // We need to make one slice per point here in case we need to move
      // points to remote nodes. The way we do slicing right now prevents
      // us from knowing which point tasks are going remote until later in
      // the replay so we have to be pessimistic here
      new_slice->expand_replay_slices(slices);
      // Then do the replay on all the slices
      for (std::list<SliceTask*>::const_iterator it = 
            slices.begin(); it != slices.end(); it++)
        (*it)->trigger_replay();
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_mapped(Deserializer &derez,
                                                    AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_mapped(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_complete(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_complete(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_commit(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      IndexTask *task;
      derez.deserialize(task);
      task->unpack_slice_commit(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_find_intra_dependence(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *task;
      derez.deserialize(task);
      DomainPoint point;
      derez.deserialize(point);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      const RtEvent result = task->find_intra_space_dependence(point);
      Runtime::trigger_event(to_trigger, result);
    }

    //--------------------------------------------------------------------------
    /*static*/ void IndexTask::process_slice_record_intra_dependence(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *task;
      derez.deserialize(task);
      DomainPoint point, next;
      derez.deserialize(point);
      derez.deserialize(next);
      RtEvent mapped_event;
      derez.deserialize(mapped_event);
      task->record_intra_space_dependence(point, next, mapped_event);
    }

#ifdef DEBUG_LEGION
    //--------------------------------------------------------------------------
    void IndexTask::check_point_requirements(
            const std::map<DomainPoint,std::vector<LogicalRegion> > &point_reqs)
    //--------------------------------------------------------------------------
    {
      // Need to run this if we haven't run it yet in order to populate
      // the interfering_requirements data structure
      perform_intra_task_alias_analysis();
      std::set<std::pair<unsigned,unsigned> > local_interfering = 
        interfering_requirements;
      // Handle any region requirements that interfere with itself
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const RegionRequirement &req = regions[idx];
        if (!IS_WRITE(req))
          continue;
        // If the projection functions are invertible then we don't have to 
        // worry about interference because the runtime knows how to hook
        // up those kinds of dependences
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
        {
          ProjectionFunction *func = 
            runtime->find_projection_function(req.projection);   
          if (func->is_invertible)
            continue;
        }
        local_interfering.insert(std::pair<unsigned,unsigned>(idx,idx));
      }
      // Nothing to do if there are no interfering requirements
      if (local_interfering.empty())
        return;
      // Make sure that all the slices coming back here are serialized
      AutoLock o_lock(op_lock);
      for (std::map<DomainPoint,std::vector<LogicalRegion> >::const_iterator 
            pit = point_reqs.begin(); pit != point_reqs.end(); pit++)
      { 
        const std::vector<LogicalRegion> &point_reqs = pit->second;
        for (std::map<DomainPoint,std::vector<LogicalRegion> >::const_iterator
              oit = point_requirements.begin(); 
              oit != point_requirements.end(); oit++)
        {
          const std::vector<LogicalRegion> &other_reqs = oit->second;
          const bool same_point = (pit->first == oit->first);
          // Now check for interference with any other points
          for (std::set<std::pair<unsigned,unsigned> >::const_iterator it =
                local_interfering.begin(); it !=
                local_interfering.end(); it++)
          {
            // Skip same region requireemnt for same point
            if (same_point && (it->first == it->second))
              continue;
            // If either one are the NO_REGION then there is no interference
            if (!point_reqs[it->first].exists() || 
                !other_reqs[it->second].exists())
              continue;
            // If the user marked this region requirement as collective
            // and this is the same region requirement for both points
            // and the region name is the same then we allow that
            if (!same_point && (it->first == it->second) &&
                IS_COLLECTIVE(regions[it->first]) &&
                (point_reqs[it->first] == other_reqs[it->second]))
              continue;
            if (!runtime->forest->are_disjoint(
                  point_reqs[it->first].get_index_space(), 
                  other_reqs[it->second].get_index_space()))
            {
              switch (pit->first.get_dim())
              {
                case 1:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point %lld and region "
                              "requirement %d of point %lld of %s (UID %lld) "
                              "in parent task %s (UID %lld) are interfering.",
                              it->first, pit->first[0], it->second,
                              oit->first[0], get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#if LEGION_MAX_DIM >= 2
                case 2:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point (%lld,%lld) and "
                              "region requirement %d of point (%lld,%lld) of "
                              "%s (UID %lld) in parent task %s (UID %lld) are "
                              "interfering.", it->first, pit->first[0],
                              pit->first[1], it->second, oit->first[0],
                              oit->first[1], get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 3
                case 3:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point (%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld) of %s (UID %lld) in parent "
                              "task %s (UID %lld) are interfering.", it->first,
                              pit->first[0], pit->first[1], pit->first[2],
                              it->second, oit->first[0], oit->first[1],
                              oit->first[2], get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 4
                case 4:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld) of %s (UID %lld) in parent"
                              " task %s (UID %lld) are interfering.", it->first,
                              pit->first[0], pit->first[1], pit->first[2],
                              pit->first[3], it->second, oit->first[0], 
                              oit->first[1], oit->first[2], oit->first[3],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 5
                case 5:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld) of %s (UID %lld) "
                              "in parent task %s (UID %lld) are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              it->second, oit->first[0], oit->first[1], 
                              oit->first[2], oit->first[3], oit->first[4],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 6
                case 6:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld) of %s " 
                              "(UID %lld) in parent task %s (UID %lld) "
                              "are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              pit->first[5], it->second, oit->first[0], 
                              oit->first[1], oit->first[2], oit->first[3], 
                              oit->first[4], oit->first[5],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 7
                case 7:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld) of %s "
                              "(UID %lld) in parent task %s (UID %lld) "
                              "are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              pit->first[5], pit->first[6], it->second, 
                              oit->first[0], oit->first[1], oit->first[2], 
                              oit->first[3], oit->first[4], oit->first[5],
                              oit->first[6], get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 8
                case 8:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) "
                              "of %s (UID %lld) in parent task %s (UID %lld) "
                              "are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              pit->first[5], pit->first[6], pit->first[7],
                              it->second, oit->first[0], oit->first[1], 
                              oit->first[2], oit->first[3], oit->first[4], 
                              oit->first[5], oit->first[6], oit->first[7],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
#if LEGION_MAX_DIM >= 9
                case 9:
                  {
                    REPORT_LEGION_ERROR(ERROR_INDEX_SPACE_TASK,
                              "Index space task launch has intefering "
                              "region requirements %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld)"
                              " and region requirement %d of point "
                              "(%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld) "
                              "of %s (UID %lld) in parent task %s (UID %lld) "
                              "are interfering.",
                              it->first, pit->first[0], pit->first[1], 
                              pit->first[2], pit->first[3], pit->first[4],
                              pit->first[5], pit->first[6], pit->first[7],
                              pit->first[8], it->second, oit->first[0], 
                              oit->first[1], oit->first[2], oit->first[3], 
                              oit->first[4], oit->first[5], oit->first[6], 
                              oit->first[7], oit->first[8],
                              get_task_name(), get_unique_id(),
                              parent_ctx->get_task_name(),
                              parent_ctx->get_unique_id());
                    break;
                  }
#endif
                default:
                  assert(false);
              }
            }
          }
        }
        // Add it to the set of point requirements
        point_requirements.insert(*pit);
      }
    }
#endif

    /////////////////////////////////////////////////////////////
    // Slice Task 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SliceTask::SliceTask(Runtime *rt)
      : MultiTask(rt)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SliceTask::SliceTask(const SliceTask &rhs)
      : MultiTask(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    SliceTask::~SliceTask(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SliceTask& SliceTask::operator=(const SliceTask &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void SliceTask::activate(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_ACTIVATE_CALL);
      MultiTask::activate();
      num_unmapped_points = 0;
      num_uncomplete_points = 0;
      num_uncommitted_points = 0;
      index_owner = NULL;
      remote_unique_id = get_unique_id();
      origin_mapped = false;
      origin_mapped_complete = RtUserEvent::NO_RT_USER_EVENT;
      // Slice tasks always already have their options selected
      options_selected = true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_DEACTIVATE_CALL);
      MultiTask::deactivate(false/*free*/);
      // Deactivate all our points 
      for (std::vector<PointTask*>::const_iterator it = points.begin();
            it != points.end(); it++)
      {
        // Check to see if we are origin mapped or not which 
        // determines whether we should commit this operation or
        // just deactivate it like normal
        if (is_origin_mapped() && !is_remote())
          (*it)->deactivate();
        else
          (*it)->commit_operation(true/*deactivate*/);
      }
      points.clear(); 
#ifdef DEBUG_LEGION
      assert(local_regions.empty());
      assert(local_fields.empty());
      assert(!origin_mapped_complete.exists() ||
              origin_mapped_complete.has_triggered());
#endif
      map_applied_conditions.clear();
      point_completions.clear();
      complete_preconditions.clear();
      commit_preconditions.clear();
      created_regions.clear();
      created_fields.clear();
      created_field_spaces.clear();
      created_index_spaces.clear();
      created_index_partitions.clear();
      unique_intra_space_deps.clear();
      concurrent_points.clear();
      if (freeop)
        runtime->free_slice_task(this);
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void SliceTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    void SliceTask::premap_task(void)
    //--------------------------------------------------------------------------
    {
      // Slices are already done with early mapping 
    }

    //--------------------------------------------------------------------------
    void SliceTask::check_target_processors(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!points.empty());
#endif
      if (points.size() == 1)
        return;
      const AddressSpaceID target_space = 
        runtime->find_address_space(points[0]->target_proc);
      for (unsigned idx = 1; idx < points.size(); idx++)
      {
        if (target_space != 
            runtime->find_address_space(points[idx]->target_proc))
          REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_OUTPUT,
                      "Invalid mapper output: two different points in one "
                      "slice of %s (UID %lld) mapped to processors in two"
                      "different address spaces (%d and %d) which is illegal.",
                      get_task_name(), get_unique_id(), target_space,
                      runtime->find_address_space(points[idx]->target_proc))
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::update_target_processor(void)
    //--------------------------------------------------------------------------
    {
      if (points.empty())
        return;
#ifdef DEBUG_LEGION
      check_target_processors();
#endif
      this->target_proc = points[0]->target_proc;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_DISTRIBUTE_CALL);
      update_target_processor();
      if (target_proc.exists() && (target_proc != current_proc))
      {
        runtime->send_task(this);
        // The runtime will deactivate this task
        // after it has been sent
        return false;
      }
      return true;
    }

    //--------------------------------------------------------------------------
    VersionInfo& SliceTask::get_version_info(unsigned idx)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        return TaskOp::get_version_info(idx);
      else
        return index_owner->get_version_info(idx);
    }

    //--------------------------------------------------------------------------
    const VersionInfo& SliceTask::get_version_info(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        return TaskOp::get_version_info(idx);
      else
        return index_owner->get_version_info(idx);
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::perform_mapping(MustEpochOp *epoch_owner/*=NULL*/,
                                       const DeferMappingArgs *args/*=NULL*/)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_PERFORM_MAPPING_CALL);
#ifdef DEBUG_LEGION
      // Should never get duplicate invocations here
      assert(args == NULL);
#endif
      // Check to see if we already enumerated all the points, if
      // not then do so now
      if (points.empty())
        enumerate_points(false/*inlining*/);
      // Once we start mapping then we are no longer stealable
      stealable = false;
      std::vector<RtEvent> mapped_events;
      for (std::vector<PointTask*>::const_iterator it = 
            points.begin(); it != points.end(); it++)
      {
        // Now that we support collective instance creation, we need to 
        // enable all the point tasks to be mapping in parallel with
        // each other in case they need to synchronize to create 
        // collective instances
        const RtEvent point_mapped = concurrent_task ?
          (*it)->defer_perform_mapping(RtEvent::NO_RT_EVENT,
              epoch_owner, args, 0/*invocation count*/) :
          (*it)->perform_mapping(epoch_owner);
        if (point_mapped.exists())
          mapped_events.push_back(point_mapped);
      }
      return Runtime::merge_events(mapped_events);
    }

    //--------------------------------------------------------------------------
    void SliceTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_LAUNCH_CALL);
#ifdef DEBUG_LEGION
      assert(!points.empty());
#endif
      // Launch all of our child points
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->launch_task(inline_task);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_origin) && stealable);
    }

    //--------------------------------------------------------------------------
    void SliceTask::map_and_launch(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_MAP_AND_LAUNCH_CALL);
      // First enumerate all of our points if we haven't already done so
      if (points.empty())
        enumerate_points(false/*inlining*/);
      // Mark that this task is no longer stealable.  Once we start
      // executing things onto a specific processor slices cannot move.
      stealable = false;
#ifdef DEBUG_LEGION
      assert(!points.empty());
#endif
      const size_t num_points = points.size();
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        PointTask *point = points[idx];
        // Now that we support collective instance creation, we need to 
        // enable all the point tasks to be mapping in parallel with
        // each other in case they need to synchronize to create 
        // collective instances
        const RtEvent point_mapped = concurrent_task ?
          point->defer_perform_mapping(RtEvent::NO_RT_EVENT,
              NULL/*must epoch*/, NULL/*defer args*/, 0/*invocation count*/) :
          point->perform_mapping();
        if (point_mapped.exists() && !point_mapped.has_triggered())
          point->defer_launch_task(point_mapped);
        else
          point->launch_task();
      }
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_output_global(unsigned idx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < output_region_options.size());
#endif
      return output_region_options[idx].global_indexing();
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_output_valid(unsigned idx) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(idx < output_region_options.size());
#endif
      return output_region_options[idx].valid_requirement();
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind SliceTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return SLICE_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::pack_task(Serializer &rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_PACK_TASK_CALL);
      // Check to see if we are stealable or not yet fully sliced,
      // if both are false and we're not remote, then we can send the state
      // now or check to see if we are remotely mapped
      RezCheck z(rez);
      // Preamble used in TaskOp::unpack
      rez.serialize(points.size());
      pack_multi_task(rez, target);
      rez.serialize(index_owner);
      rez.serialize(remote_unique_id);
      rez.serialize(origin_mapped);
      parent_ctx->pack_inner_context(rez);
      rez.serialize(internal_space);
      if (!elide_future_return)
      {
        if (redop == 0)
        {
#ifdef DEBUG_LEGION
          assert(future_map.impl != NULL);
#endif
          future_map.impl->pack_future_map(rez, target);
        }
        if (predicate_false_future.impl != NULL)
          predicate_false_future.impl->pack_future(rez, target);
        else
          rez.serialize<DistributedID>(0);
        rez.serialize(predicate_false_size);
        if (predicate_false_size > 0)
          rez.serialize(predicate_false_result, predicate_false_size);
      }
      Provenance *provenance = get_provenance();
      if (provenance != NULL)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->pack_task(rez, target);
      // If we don't have any points, we have to pack up the argument map
      // and any trace info that we need for doing remote tracing
      if (points.empty())
      {
        if (point_arguments.impl != NULL)
          point_arguments.impl->pack_future_map(rez, target);
        else
          rez.serialize<DistributedID>(0);
        rez.serialize<size_t>(point_futures.size());
        for (unsigned idx = 0; idx < point_futures.size(); idx++)
        {
          FutureMapImpl *impl = point_futures[idx].impl;
          impl->pack_future_map(rez, target);
        }
      }
      if (is_origin_mapped() && !is_remote())
      {
#ifdef DEBUG_LEGION
        assert(!origin_mapped_complete.exists());
#endif
        // Similarly for slices being removed remotely but are
        // origin mapped we may need to receive profiling feedback
        // to this node so also hold onto these slices until the
        // index space is done
        index_owner->record_origin_mapped_slice(this);
        return false;
      }
      // Always return true for slice tasks since they should
      // always be deactivated after they are sent somewhere else
      return true;
    }
    
    //--------------------------------------------------------------------------
    bool SliceTask::unpack_task(Deserializer &derez, Processor current,
                                std::set<RtEvent> &ready_events)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_UNPACK_TASK_CALL);
      DerezCheck z(derez);
      size_t num_points;
      derez.deserialize(num_points);
      unpack_multi_task(derez, ready_events);
      set_current_proc(current);
      derez.deserialize(index_owner);
      derez.deserialize(remote_unique_id); 
      derez.deserialize(origin_mapped);
      parent_ctx = InnerContext::unpack_inner_context(derez, runtime);
      derez.deserialize(internal_space);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_slice_slice(remote_unique_id, get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(remote_unique_id,
            get_unique_op_id());
      num_unmapped_points = num_points;
      num_uncomplete_points = num_points;
      num_uncommitted_points = num_points;
      // Remote slice tasks are always resolved
      resolved = true;
      if (!elide_future_return)
      {
        if (redop == 0)
          future_map = FutureMapImpl::unpack_future_map(runtime, derez, 
                                                        parent_ctx);
        // Unpack the predicate false infos
        predicate_false_future = FutureImpl::unpack_future(runtime, derez);
        derez.deserialize(predicate_false_size);
        if (predicate_false_size > 0)
        {
#ifdef DEBUG_LEGION
          assert(predicate_false_result == NULL);
#endif
          predicate_false_result = malloc(predicate_false_size);
          derez.deserialize(predicate_false_result, predicate_false_size);
        }
      }
      // Unpack the provenance before unpacking any point tasks so
      // that they can pick it up as well
      set_provenance(Provenance::deserialize(derez));
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        PointTask *point = runtime->get_available_point_task(); 
        point->slice_owner = this;
        point->unpack_task(derez, current, ready_events);
        point->parent_ctx = parent_ctx;
        points.push_back(point);
        if (runtime->legion_spy_enabled)
          LegionSpy::log_slice_point(get_unique_id(), 
                                     point->get_unique_id(),
                                     point->index_point);
      }
      if (num_points == 0)
      {
        point_arguments = FutureMapImpl::unpack_future_map(runtime, derez, 
                                                           parent_ctx);
        size_t num_point_futures;
        derez.deserialize(num_point_futures);
        if (num_point_futures > 0)
        {
          point_futures.resize(num_point_futures);
          for (unsigned idx = 0; idx < num_point_futures; idx++)
            point_futures[idx] = FutureMapImpl::unpack_future_map(runtime, 
                                                        derez, parent_ctx);
        }
      }
      else // Set the first mapping to false since we know things are mapped
        first_mapping = false;
      if (runtime->profiler != NULL)
        runtime->profiler->register_operation(this);
      // Return true to add this to the ready queue
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::perform_inlining(VariantImpl *variant,
                                const std::deque<InstanceSet> &parent_instances)
    //--------------------------------------------------------------------------
    {
      // Need to handle inter-space dependences correctly here
      std::map<PointTask*,unsigned> remaining;
      std::map<RtEvent,std::vector<PointTask*> > event_deps;
      for (std::vector<PointTask*>::const_iterator it =
            points.begin(); it != points.end(); it++)
        if (!(*it)->has_remaining_inlining_dependences(remaining, event_deps))
          (*it)->perform_inlining(variant, parent_instances);
      while (!remaining.empty())
      {
#ifdef DEBUG_LEGION
        bool found = false; // should find at least one each iteration
#endif
        for (std::map<PointTask*,unsigned>::iterator it =
              remaining.begin(); it != remaining.end(); /*nothing*/)
        {
          if (it->second == 0)
          {
            const RtEvent mapped = it->first->get_mapped_event();
            it->first->perform_inlining(variant, parent_instances);
#ifdef DEBUG_LEGION
            found = true;
            assert(mapped.has_triggered());
#endif
            std::map<RtEvent,std::vector<PointTask*> >::const_iterator finder =
              event_deps.find(mapped);
            if (finder != event_deps.end())
            {
              for (unsigned idx = 0; idx < finder->second.size(); idx++)
              {
                std::map<PointTask*,unsigned>::iterator point_finder =
                  remaining.find(finder->second[idx]);
#ifdef DEBUG_LEGION
                assert(point_finder != remaining.end());
                assert(point_finder->second > 0);
#endif
                point_finder->second--;
              }
              event_deps.erase(finder);
            }
            std::map<PointTask*,unsigned>::iterator to_delete = it++;
            remaining.erase(to_delete);
          }
          else
            it++;
        }
#ifdef DEBUG_LEGION
        assert(found);
#endif
      }
      // Record that we've mapped and executed this slice
      trigger_slice_mapped();
    } 

    //--------------------------------------------------------------------------
    SliceTask* SliceTask::clone_as_slice_task(IndexSpace is, Processor p,
                                              bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_CLONE_AS_SLICE_CALL);
      SliceTask *result = runtime->get_available_slice_task(); 
      result->initialize_base_task(parent_ctx, Predicate::TRUE_PRED,
                                   this->task_id, get_provenance());
      result->clone_multi_from(this, is, p, recurse, stealable);
      result->index_owner = this->index_owner;
      if (runtime->legion_spy_enabled)
        LegionSpy::log_slice_slice(get_unique_id(), 
                                   result->get_unique_id());
      if (runtime->profiler != NULL)
        runtime->profiler->register_slice_owner(get_unique_op_id(),
            result->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::reduce_future(const DomainPoint &point,
                                  FutureInstance *inst, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        // Store the future result in our temporary futures unless we're 
        // doing a non-deterministic reduction in which case we can eagerly
        // fold this now into our reduction buffer
        if (deterministic_redop)
        {
          // Store it in our temporary futures
          // Hold the lock to protect the data structure
          AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
          assert(temporary_futures.find(point) == temporary_futures.end());
#endif
          temporary_futures[point] = std::make_pair(inst, effects);
        }
        else
        {
          // If we're not doing serdez functions, we'll grab the first
          // one of these instances as the target for us to reduce into
          if ((serdez_redop_fns == NULL) && (reduction_instance == NULL))
          {
            AutoLock o_lock(op_lock);
            // See if we lost the race
            if (reduction_instance == NULL)
            {
              reduction_instance_point = point;
              if (inst->is_meta_visible ||
                  (inst->size > LEGION_MAX_RETURN_SIZE))
              {
                reduction_instance_precondition = effects;
                // Must be the last thing we store
                reduction_instance = inst;
                return;
              }
              else
                reduction_instance =
                  FutureInstance::create_local(&reduction_op->identity,
                      reduction_op->sizeof_rhs, false/*own*/);
            }
          }
          if (!fold_reduction_future(inst, effects))
          {
            // save it to delete later
            AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
            assert(temporary_futures.find(point) == temporary_futures.end());
#endif
            temporary_futures[point] = std::make_pair(inst, effects);
          }
          else
            delete inst;
        }
      }
      else
        index_owner->reduce_future(point, inst, effects); 
    }

    //--------------------------------------------------------------------------
    void SliceTask::handle_future(ApEvent effects, const DomainPoint &point,
                                  FutureInstance *instance,
                                  void *metadata, size_t metasize,
                                  FutureFunctor *functor,
                                  Processor future_proc, bool own_functor)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_HANDLE_FUTURE_CALL);
      if (elide_future_return)
      {
        if (functor != NULL)
        {
#ifdef DEBUG_LEGION
          assert(instance == NULL);
          assert(metadata == NULL);
#endif
          functor->callback_release_future();
          if (own_functor)
            delete functor;
        }
        else if ((instance != NULL) && !instance->defer_deletion(effects))
          delete instance;
      }
      else if (redop > 0)
      {
#ifdef DEBUG_LEGION
        assert(functor == NULL);
        assert(instance != NULL);
#endif
        reduce_future(point, instance, effects);
        if (metadata != NULL)
        {
          AutoLock o_lock(op_lock);
          if (reduction_metadata == NULL)
          {
            reduction_metadata = metadata;
            reduction_metasize = metasize;
            metadata = NULL; // we took ownership of allocation
          }
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(future_handles != NULL);
#endif
        std::map<DomainPoint,DistributedID>::const_iterator finder = 
          future_handles->handles.find(point);
#ifdef DEBUG_LEGION
        assert(finder != future_handles->handles.end());
#endif
        const ContextCoordinate coordinate(future_map_coordinate, point);
        FutureImpl *impl = runtime->find_or_create_future(finder->second, 
            parent_ctx->did, coordinate, get_provenance());
        if (functor != NULL)
        {
#ifdef DEBUG_LEGION
          assert(instance == NULL);
          assert(metadata == NULL);
#endif
          impl->set_result(effects, functor, own_functor, future_proc);
        }
        else
        {
          impl->set_result(effects, instance, metadata, metasize);
          metadata = NULL; // no longer own the allocation
        }
      }
      if (metadata != NULL)
        free(metadata);
    }

    //--------------------------------------------------------------------------
    void SliceTask::register_must_epoch(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(must_epoch != NULL);
#endif
      if (points.empty())
        enumerate_points(false/*inling*/);
      must_epoch->register_slice_task(this);
      for (unsigned idx = 0; idx < points.size(); idx++)
      {
        PointTask *point = points[idx];
        must_epoch->register_single_task(point, must_epoch_index);
      }
    }

    //--------------------------------------------------------------------------
    PointTask* SliceTask::clone_as_point_task(const DomainPoint &point,
                                              bool inline_task)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_CLONE_AS_POINT_CALL);
      PointTask *result = runtime->get_available_point_task();
      result->initialize_base_task(parent_ctx, Predicate::TRUE_PRED,
                                   this->task_id, get_provenance());
      result->clone_task_op_from(this, this->target_proc, 
                                 false/*stealable*/, true/*duplicate*/);
      result->is_index_space = true;
      result->must_epoch_task = this->must_epoch_task;
      result->index_domain = this->index_domain;
      result->version_infos.resize(logical_regions.size());
      // Now figure out our local point information
      result->initialize_point(this, point, point_arguments,
                               inline_task, point_futures);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_slice_point(get_unique_id(), 
                                   result->get_unique_id(),
                                   result->index_point);
      return result;
    }

    //--------------------------------------------------------------------------
    size_t SliceTask::enumerate_points(bool inline_task)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_ENUMERATE_POINTS_CALL);
      Domain internal_domain;
      runtime->forest->find_domain(internal_space, internal_domain);
      const size_t num_points = internal_domain.get_volume();
#ifdef DEBUG_LEGION
      assert(num_points > 0);
#endif
      unsigned point_idx = 0;
      points.resize(num_points);
      // Enumerate all the points in our slice and make point tasks
      for (Domain::DomainPointIterator itr(internal_domain); 
            itr; itr++, point_idx++)
        points[point_idx] = clone_as_point_task(itr.p, inline_task);
      // Compute any projection region requirements
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
      {
        const RegionRequirement &req = logical_regions[idx];
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
          continue;
        ProjectionFunction *function = 
          runtime->find_projection_function(req.projection);
        function->project_points(req, idx, runtime, index_domain, points);
      }
      // Update the no access regions
      for (unsigned idx = 0; idx < num_points; idx++)
        points[idx]->complete_point_projection();
      // Mark how many points we have
      num_unmapped_points = num_points;
      num_uncomplete_points = num_points;
      num_uncommitted_points = num_points;
      return num_points;
    } 

    //--------------------------------------------------------------------------
    void SliceTask::set_predicate_false_result(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      if (elide_future_return || (redop > 0))
        return;
#ifdef DEBUG_LEGION
      assert(future_handles != NULL);
#endif
      std::map<DomainPoint,DistributedID>::const_iterator finder = 
        future_handles->handles.find(point);
#ifdef DEBUG_LEGION
      assert(finder != future_handles->handles.end());
#endif
      const ContextCoordinate coordinate(future_map_coordinate, point);
      FutureImpl *impl = runtime->find_or_create_future(finder->second, 
          parent_ctx->did, coordinate, get_provenance());
      if (predicate_false_future.impl == NULL)
      {
        if (predicate_false_size > 0)
          impl->set_local(predicate_false_result,
                                 predicate_false_size, false/*own*/);
        else
          impl->set_result(ApEvent::NO_AP_EVENT, NULL);
      }
      else
        impl->set_result(predicate_false_future.impl, this);
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_complete(void)
    //--------------------------------------------------------------------------
    {
      // This is mostly the same as the base TaskOp::trigger_complete
      // except we also check to see if we have an origin_mapped_complete
      // event that we need to trigger once we're done
      RtUserEvent to_trigger;
      bool task_complete = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(!complete_received);
        assert(!commit_received);
#endif
        complete_received = true;
        // If all our children are also complete then we are done
        task_complete = children_complete;
        if (origin_mapped_complete.exists())
          to_trigger = origin_mapped_complete;
      }
      if (task_complete)
        trigger_task_complete();
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_task_complete(void)
    //--------------------------------------------------------------------------
    {
      trigger_slice_complete();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      trigger_slice_commit();
    } 

    //--------------------------------------------------------------------------
    void SliceTask::record_completion_effect(ApEvent effect)
    //--------------------------------------------------------------------------
    {
      if (!effect.exists())
        return;
      if (is_remote())
      {
        AutoLock o_lock(op_lock);
        point_completions.insert(effect);
      }
      else
        index_owner->record_completion_effect(effect);
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_completion_effect(ApEvent effect,
                                          std::set<RtEvent> &map_applied_events)
    //--------------------------------------------------------------------------
    {
      if (!effect.exists())
        return;
      if (is_remote())
      {
        AutoLock o_lock(op_lock);
        point_completions.insert(effect);
      }
      else
        index_owner->record_completion_effect(effect, map_applied_events);
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_completion_effects(const std::set<ApEvent> &effects)
    //--------------------------------------------------------------------------
    {
      if (effects.empty())
        return;
      if (is_remote())
      {
        AutoLock o_lock(op_lock);
        for (std::set<ApEvent>::const_iterator it =
              effects.begin(); it != effects.end(); it++)
          if (it->exists())
            point_completions.insert(*it);
      }
      else
        index_owner->record_completion_effects(effects);
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_completion_effects(
                                            const std::vector<ApEvent> &effects)
    //--------------------------------------------------------------------------
    {
      if (effects.empty())
        return;
      if (is_remote())
      {
        AutoLock o_lock(op_lock);
        for (std::vector<ApEvent>::const_iterator it =
              effects.begin(); it != effects.end(); it++)
          if (it->exists())
            point_completions.insert(*it);
      }
      else
        index_owner->record_completion_effects(effects);
    }

    //--------------------------------------------------------------------------
    void SliceTask::return_privileges(TaskContext *point_context,
                                      std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // If we're remote, pass our privileges back to ourself
      // otherwise pass them directly back to the index owner
      if (is_remote())
        point_context->return_resources(this, context_index, preconditions);
      else if (must_epoch != NULL)
        point_context->return_resources(must_epoch, context_index,
                                        preconditions);
      else
        point_context->return_resources(parent_ctx, context_index,
                                        preconditions);
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_point_mapped(RtEvent child_mapped)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (child_mapped.exists())
          map_applied_conditions.insert(child_mapped);
#ifdef DEBUG_LEGION
        assert(num_unmapped_points > 0);
#endif
        num_unmapped_points--;
        if (num_unmapped_points == 0)
          needs_trigger = true;
      }
      if (needs_trigger)
        trigger_slice_mapped();
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_point_complete(RtEvent child_complete)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (child_complete.exists())
          complete_preconditions.insert(child_complete);
#ifdef DEBUG_LEGION
        assert(num_uncomplete_points > 0);
#endif
        num_uncomplete_points--;
        if ((num_uncomplete_points == 0) && !children_complete_invoked)
        {
          needs_trigger = true;
          children_complete_invoked = true;
        }
      }
      if (needs_trigger)
        trigger_children_complete();
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_point_committed(RtEvent commit_precondition)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(num_uncommitted_points > 0);
#endif
        if (commit_precondition.exists())
          commit_preconditions.insert(commit_precondition);
        num_uncommitted_points--;
        if ((num_uncommitted_points == 0) && !children_commit_invoked)
        {
          needs_trigger = true;
          children_commit_invoked = true;
        }
      }
      if (needs_trigger)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void SliceTask::handle_future_size(size_t future_size,
                const DomainPoint &point, std::set<RtEvent> &applied_conditions)
    //--------------------------------------------------------------------------
    {
      if (redop > 0)
        return;
#ifdef DEBUG_LEGION
      assert(!elide_future_return);
      assert(future_handles != NULL);
#endif
      const std::map<DomainPoint,DistributedID> &handles = 
        future_handles->handles;
      std::map<DomainPoint,DistributedID>::const_iterator finder = 
        handles.find(point);
#ifdef DEBUG_LEGION
      assert(finder != handles.end());
#endif
      const ContextCoordinate coordinate(future_map_coordinate, point);
      FutureImpl *impl = runtime->find_or_create_future(finder->second, 
        parent_ctx->did, coordinate, get_provenance());
      impl->set_future_result_size(future_size, runtime->address_space);
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_output_extent(unsigned index,
                            const DomainPoint &color, const DomainPoint &extent)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(index < output_regions.size());
      assert(output_regions.size() == output_region_extents.size());
      assert(output_regions.size() == output_region_options.size());
      assert(!is_output_valid(index));
#endif
      {
        AutoLock o_lock(op_lock);
        OutputExtentMap &output_extents = output_region_extents[index];
        if (output_extents.find(color) != output_extents.end())
        {
          const OutputRequirement &req = output_regions[index];
          std::stringstream ss;
          ss << "(" << color[0];
          for (int dim = 1; dim < color.dim; ++dim) ss << "," << color[dim];
          ss << ")";
          REPORT_LEGION_ERROR(ERROR_INVALID_OUTPUT_REGION_PROJECTION,
            "A projection functor for every output requirement must be "
            "bijective, but projection functor %u for output requirement %u "
            "in task %s (UID: %lld) mapped more than one point in the launch "
            "domain to the same subregion of color %s.",
            req.projection, index, get_task_name(), get_unique_op_id(),
            ss.str().c_str());
        }
        output_extents[color] = extent;
#ifdef DEBUG_LEGION
        assert(output_extents.size() <= points.size());
#endif
        if (output_extents.size() < points.size())
          return;
        // Check the other output regions to see if they are done as well
        for (unsigned idx = 0; idx < output_regions.size(); idx++)
        {
          if (idx == index)
            continue;
          if (is_output_valid(idx))
            continue;
#ifdef DEBUG_LEGION
          assert(output_region_extents[idx].size() <= points.size());
#endif
          if (output_region_extents[idx].size() < points.size())
            return;
        }
      }
      // If we get here then we need to send the sizes back to the index owner
      if (is_remote())
      {
        const RtUserEvent applied = Runtime::create_rt_user_event();
        // Send a message back to the owner with the output region extents
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize<size_t>(output_region_extents.size());
          for (unsigned idx = 0; idx < output_region_extents.size(); idx++)
          {
            const OutputExtentMap &extents = output_region_extents[idx];
            rez.serialize<size_t>(extents.size());
            for (OutputExtentMap::const_iterator it =
                  extents.begin(); it != extents.end(); it++)
            {
              rez.serialize(it->first);
              rez.serialize(it->second);
            }
          }
          rez.serialize(applied);
        }
        runtime->send_slice_remote_output_extents(orig_proc, rez);
        AutoLock o_lock(op_lock);
#ifdef DEBUG_LEGION
        assert(num_uncomplete_points > 0);
#endif
        complete_preconditions.insert(applied);
      }
      else
        index_owner->record_output_extents(output_region_extents);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_remote_output_extents(Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *index_owner;
      derez.deserialize(index_owner);
      size_t num_regions;
      derez.deserialize(num_regions);
      std::vector<OutputExtentMap> output_region_extents(num_regions);
      for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
      {
        OutputExtentMap &extents = output_region_extents[idx1];
        size_t num_extents;
        derez.deserialize(num_extents);
        for (unsigned idx2 = 0; idx2 < num_extents; idx2++)
        {
          DomainPoint color;
          derez.deserialize(color);
          derez.deserialize(extents[color]);
        }
      }
      index_owner->record_output_extents(output_region_extents);
      RtUserEvent applied;
      derez.deserialize(applied);
      Runtime::trigger_event(applied);
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_output_registered(RtEvent registered,
                                             std::set<RtEvent> &applied_events)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(registered.exists());
#endif
      if (is_remote())
      {
        // Send a message back to the index owner about the equivalence
        // sets for the output regions being registered
        const RtUserEvent applied = Runtime::create_rt_user_event();
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize(registered);
          rez.serialize(applied);
        }
        runtime->send_slice_remote_output_registration(orig_proc, rez);
        applied_events.insert(applied);
      }
      else
        index_owner->record_output_registered(registered);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_remote_output_registration(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *index_owner;
      derez.deserialize(index_owner);
      RtEvent registered;
      derez.deserialize(registered);
      RtUserEvent applied;
      derez.deserialize(applied);
      index_owner->record_output_registered(registered);
      Runtime::trigger_event(applied);
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::verify_concurrent_execution(const DomainPoint &point,
                                                   Processor target)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(concurrent_task);
#endif
      if (is_remote())
      {
        AutoLock o_lock(op_lock);
        if (concurrent_processors.empty())
        {
#ifdef DEBUG_LEGION
          assert(!concurrent_verified.exists());
#endif
          concurrent_verified = Runtime::create_rt_user_event();
        }
#ifdef DEBUG_LEGION
        assert(concurrent_processors.find(point) == 
                concurrent_processors.end());
        assert(concurrent_processors.size() < points.size());
#endif
        concurrent_processors[point] = target;
        if (concurrent_processors.size() == points.size())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(index_owner);
            rez.serialize<size_t>(points.size());
            for (std::map<DomainPoint,Processor>::const_iterator it =
                  concurrent_processors.begin(); it != 
                  concurrent_processors.end(); it++)
            {
              rez.serialize(it->first);
              rez.serialize(it->second);
            }
            rez.serialize(concurrent_verified);
          }
          runtime->send_slice_verify_concurrent_execution(orig_proc, rez);
        }
        return concurrent_verified;
      }
      else
        return index_owner->verify_concurrent_execution(point, target);
    }

    //--------------------------------------------------------------------------
    void SliceTask::concurrent_allreduce(PointTask *task, 
                            ProcessorManager *manager, uint64_t lamport_clock,
                            VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      {
        AutoLock o_lock(op_lock);
        if (concurrent_lamport_clock < lamport_clock)
          concurrent_lamport_clock = lamport_clock;
        if (poisoned)
          concurrent_poisoned = true;
        if (concurrent_points.empty())
          concurrent_variant = vid;
        else if (concurrent_variant != vid)
          concurrent_variant = std::min(concurrent_variant, vid);
        concurrent_points.push_back(std::make_pair(task, manager));
        done = (concurrent_points.size() == points.size());
      }
      if (done)
      {
        if (is_remote())
        {
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(index_owner);
            rez.serialize(this);
            rez.serialize(points.size());
            rez.serialize(concurrent_lamport_clock);
            rez.serialize(concurrent_variant);
            rez.serialize(concurrent_poisoned);
          }
          runtime->send_slice_concurrent_allreduce_request(orig_proc, rez);
        }
        else
          index_owner->concurrent_allreduce(this,runtime->address_space,
              points.size(), concurrent_lamport_clock, concurrent_variant,
              concurrent_poisoned);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::finish_concurrent_allreduce(uint64_t lamport_clock,
                     bool poisoned, VariantID vid, RtBarrier concurrent_barrier)
    //--------------------------------------------------------------------------
    {
      concurrent_task_barrier = concurrent_barrier;
      // Swap this vector onto the stack in case the slice task gets deleted
      // out from under us while we are finalizing things
      std::vector<std::pair<PointTask*,ProcessorManager*> > local_copy;
      local_copy.swap(concurrent_points);
      for (std::vector<std::pair<PointTask*,ProcessorManager*> >::const_iterator
            it = local_copy.begin(); it != local_copy.end(); it++)
        if (it->first->check_concurrent_variant(vid))
          it->second->finalize_concurrent_task_order(it->first,
              lamport_clock, poisoned);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_verify_concurrent_execution(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *owner;
      derez.deserialize(owner);
      size_t num_points;
      derez.deserialize(num_points);
      RtEvent verified;
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        Processor proc;
        derez.deserialize(proc);
        verified = owner->verify_concurrent_execution(point, proc);
      }
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done, verified);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_concurrent_allreduce_request(
                                     Deserializer &derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *owner;
      derez.deserialize(owner);
      SliceTask *slice;
      derez.deserialize(slice);
      size_t total_points;
      derez.deserialize(total_points);
      uint64_t lamport_clock;
      derez.deserialize(lamport_clock);
      VariantID variant;
      derez.deserialize(variant);
      bool poisoned;
      derez.deserialize<bool>(poisoned);
      owner->concurrent_allreduce(slice, source, total_points,
                                  lamport_clock, variant, poisoned);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_concurrent_allreduce_response(
                                                            Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      SliceTask *slice;
      derez.deserialize(slice);
      RtBarrier barrier;
      derez.deserialize(barrier);
      uint64_t lamport_clock;
      derez.deserialize(lamport_clock);
      VariantID vid;
      derez.deserialize(vid);
      bool poisoned;
      derez.deserialize<bool>(poisoned);
      slice->finish_concurrent_allreduce(lamport_clock, poisoned, vid, barrier);
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_slice_mapped(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_MAPPED_CALL);
      RtEvent applied_condition;
      if (!map_applied_conditions.empty())
        applied_condition = Runtime::merge_events(map_applied_conditions);
      ApEvent all_points_complete;
      {
        // Lock this since we could still have future reductions applied here
        AutoLock o_lock(op_lock);
        if (!point_completions.empty())
        {
          all_points_complete = Runtime::merge_events(NULL, point_completions);
          point_completions.clear();
        }
      }
      if (is_remote())
      {
        // Only need to send something back if this wasn't origin mapped 
        if (!is_origin_mapped())
        {
          Serializer rez;
          pack_remote_mapped(rez, applied_condition, all_points_complete);
          runtime->send_slice_remote_mapped(orig_proc, rez);
        }
      }
      else
      {
#ifdef DEBUG_LEGION
        // In debug mode, get all our point region requirements and
        // then pass them back to the index space task
        std::map<DomainPoint,std::vector<LogicalRegion> > local_requirements;
        for (std::vector<PointTask*>::const_iterator it = 
              points.begin(); it != points.end(); it++)
        {
          std::vector<LogicalRegion> &reqs = 
            local_requirements[(*it)->index_point];
          reqs.resize(regions.size());
          for (unsigned idx = 0; idx < regions.size(); idx++)
            reqs[idx] = (*it)->regions[idx].region;
        }
        index_owner->check_point_requirements(local_requirements);
#endif
        index_owner->return_slice_mapped(points.size(), applied_condition, 
                                         all_points_complete);
      }
      complete_mapping(applied_condition); 
      complete_execution();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_slice_complete(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_COMPLETE_CALL);
      RtEvent complete_precondition;
      if (!complete_preconditions.empty())
        complete_precondition = Runtime::merge_events(complete_preconditions);
      // For remote cases we have to keep track of the events for
      // returning any created logical state, we can't commit until
      // it is returned or we might prematurely release the references
      // that we hold on the version state objects
      if (is_remote())
      {
        // Send back the message saying that this slice is complete
        Serializer rez;
        pack_remote_complete(rez, complete_precondition);
        runtime->send_slice_remote_complete(orig_proc, rez);
      }
      else
      {
#ifdef DEBUG_LEGION
        assert(temporary_futures.empty());
        assert(reduction_instance == NULL);
        assert(serdez_redop_state == NULL);
#endif
        index_owner->return_slice_complete(points.size(), complete_precondition,
                                        reduction_metadata, reduction_metasize);
        // No longer own the buffer so clear it
        reduction_metadata = NULL;
      }
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_slice_commit(void)
    //--------------------------------------------------------------------------
    {
      DETAILED_PROFILER(runtime, SLICE_COMMIT_CALL);
      RtEvent commit_precondition;
      if (!commit_preconditions.empty())
        commit_precondition = Runtime::merge_events(commit_preconditions);
      if (is_remote())
      {
        Serializer rez;
        pack_remote_commit(rez, commit_precondition);
        runtime->send_slice_remote_commit(orig_proc, rez);
      }
      else
      {
        // created and deleted privilege information already passed back
        // futures already sent back
        index_owner->return_slice_commit(points.size(), commit_precondition);
      }
      commit_operation(true/*deactivate*/, commit_precondition);
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_mapped(Serializer &rez, 
                         RtEvent applied_condition, ApEvent all_points_complete)
    //--------------------------------------------------------------------------
    {
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize(points.size());
      rez.serialize(applied_condition);
      rez.serialize(all_points_complete);
#ifdef DEBUG_LEGION
      if (!is_origin_mapped())
      {
        for (std::vector<PointTask*>::const_iterator it = 
              points.begin(); it != points.end(); it++)
        {
          rez.serialize((*it)->index_point);
          for (unsigned idx = 0; idx < regions.size(); idx++)
            rez.serialize((*it)->regions[idx].region);
        }
      }
#endif
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_complete(Serializer &rez, 
                                         RtEvent applied_condition)
    //--------------------------------------------------------------------------
    {
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize<size_t>(points.size());
      rez.serialize(applied_condition);
      // Serialize the privilege state
      pack_resources_return(rez, context_index); 
      if (!point_completions.empty())
      {
        const ApEvent completion_effects =
          Runtime::merge_events(NULL, point_completions);
        rez.serialize(completion_effects);
        point_completions.clear();
      }
      else
        rez.serialize(ApEvent::NO_AP_EVENT);
      // Now pack up the future results
      if (redop > 0)
      {
        if (deterministic_redop)
        {
#ifdef DEBUG_LEGION
          assert(reduction_instance == NULL);
          // Might have no temporary futures if this task was predicated
          // and the predicate resolved to false
          assert((temporary_futures.size() == points.size()) || 
              temporary_futures.empty());
          assert(reduction_fold_effects.empty());
#endif
          rez.serialize<size_t>(temporary_futures.size());
          for (std::map<DomainPoint,
                std::pair<FutureInstance*,ApEvent> >::const_iterator it =
               temporary_futures.begin(); it != temporary_futures.end(); it++)
          {
            rez.serialize(it->first);
            if (!it->second.first->pack_instance(rez, it->second.second,
                                                true/*pack ownership*/))
              rez.serialize(it->second.second);
          }
        }
        else
        {
          if (serdez_redop_fns != NULL)
          {
#ifdef DEBUG_LEGION
            assert(reduction_instance == NULL);
            assert(reduction_fold_effects.empty());
#endif
            // Easy case just for serdez, we just pack up the local buffer
            rez.serialize(serdez_redop_state_size);
            if (serdez_redop_state_size > 0)
              rez.serialize(serdez_redop_state, serdez_redop_state_size);
          }
          else
          {
#ifdef DEBUG_LEGION
            // We might not have a reduction instance if this task was
            // predicated and ended up predicating false
            assert((reduction_instance != NULL) || false_guard.exists());
            assert((reduction_instance != NULL) == 
                (reduction_instance_point.get_dim() > 0));
#endif
            rez.serialize(reduction_instance_point);
            if (!reduction_fold_effects.empty())
              // All the reduction fold effects dominate the
              // reduction_instance_precondition so we can just
              // overwrite it without including it in the merger
              reduction_instance_precondition =
                Runtime::merge_events(NULL, reduction_fold_effects);
            if ((reduction_instance != NULL) &&
                !reduction_instance.load()->pack_instance(rez, 
                  reduction_instance_precondition, true/*pack ownership*/))
              rez.serialize(reduction_instance_precondition);
          }
        }
        if (reduction_metadata != NULL)
        {
          rez.serialize(reduction_metasize);
          rez.serialize(reduction_metadata, reduction_metasize);
        }
        else
          rez.serialize<size_t>(0);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_commit(Serializer &rez, 
                                       RtEvent applied_condition)
    //--------------------------------------------------------------------------
    {
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize(points.size());
      rez.serialize(applied_condition);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_slice_return(Runtime *rt, 
                                                   Deserializer &derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      RtUserEvent ready_event;
      derez.deserialize(ready_event);
      Runtime::trigger_event(ready_event);
    }

    //--------------------------------------------------------------------------
    void SliceTask::receive_resources(uint64_t return_index,
              std::map<LogicalRegion,unsigned> &created_regs,
              std::vector<DeletedRegion> &deleted_regs,
              std::set<std::pair<FieldSpace,FieldID> > &created_fids,
              std::vector<DeletedField> &deleted_fids,
              std::map<FieldSpace,unsigned> &created_fs,
              std::map<FieldSpace,std::set<LogicalRegion> > &latent_fs,
              std::vector<DeletedFieldSpace> &deleted_fs,
              std::map<IndexSpace,unsigned> &created_is,
              std::vector<DeletedIndexSpace> &deleted_is,
              std::map<IndexPartition,unsigned> &created_partitions,
              std::vector<DeletedPartition> &deleted_partitions,
              std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      merge_received_resources(created_regs, deleted_regs, created_fids, 
          deleted_fids, created_fs, latent_fs, deleted_fs, created_is,
          deleted_is, created_partitions, deleted_partitions);
    }

    //--------------------------------------------------------------------------
    void SliceTask::expand_replay_slices(std::list<SliceTask*> &slices)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!points.empty());
      assert(is_origin_mapped());
#endif
      // For each point give it its own slice owner in case we need to
      // to move it remotely as part of the replay
      while (points.size() > 1)
      {
        PointTask *point = points.back();
        points.pop_back();
        SliceTask *new_owner = clone_as_slice_task(internal_space,
                current_proc, false/*recurse*/, false/*stealable*/);
        point->slice_owner = new_owner;
        new_owner->points.push_back(point);
        new_owner->num_unmapped_points = 1;
        new_owner->num_uncomplete_points = 1;
        new_owner->num_uncommitted_points = 1;
        slices.push_back(new_owner);
      }
      // Always add ourselves as the last point
      slices.push_back(this);
      num_unmapped_points = points.size();
      num_uncomplete_points = points.size();
      num_uncommitted_points = points.size();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->trigger_replay();
    }

    //--------------------------------------------------------------------------
    void SliceTask::complete_replay(ApEvent instance_ready_event,
                                    ApEvent postcondition)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < points.size(); idx++)
        points[idx]->complete_replay(instance_ready_event, postcondition);
    }

    //--------------------------------------------------------------------------
    void SliceTask::find_commit_preconditions(std::set<RtEvent> &preconditions)
    //--------------------------------------------------------------------------
    {
      // See if any of our point tasks have any profiling report events
      for (std::vector<PointTask*>::const_iterator it = 
            points.begin(); it != points.end(); it++)
      {
        const RtEvent profiling_reported = (*it)->get_profiling_reported();
        if (profiling_reported.exists())
          preconditions.insert(profiling_reported);
      }
      // See if we haven't completed it yet, if so make an event to track it
      AutoLock o_lock(op_lock);
      if (!complete_received)
      {
        origin_mapped_complete = Runtime::create_rt_user_event();
        preconditions.insert(origin_mapped_complete);
      }
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::find_intra_space_dependence(const DomainPoint &point)
    //--------------------------------------------------------------------------
    {
      // See if we can find or make it
      {
        AutoLock o_lock(op_lock);
        std::map<DomainPoint,RtEvent>::const_iterator finder = 
          intra_space_dependences.find(point);
        // If we've already got it then we're done
        if (finder != intra_space_dependences.end())
          return finder->second;
#ifdef DEBUG_LEGION
        assert(!points.empty());
#endif
        // Next see if it is one of our local points
        for (std::vector<PointTask*>::const_iterator it = 
              points.begin(); it != points.end(); it++)
        {
          if ((*it)->index_point != point)
            continue;
          // Don't save this in our intra_space_dependences data structure!
          // Doing so could mess up our optimization for detecting when 
          // we need to send dependences back to the origin
          // See SliceTask::record_intra_space_dependence
          return (*it)->get_mapped_event();
        }
        // If we're remote, make up an event and send a message to go find it
        if (is_remote())
        {
          const RtUserEvent temp_event = Runtime::create_rt_user_event();
          // Send the message to the owner to go find it
          Serializer rez;
          {
            RezCheck z(rez);
            rez.serialize(index_owner);
            rez.serialize(point);
            rez.serialize(temp_event);
          }
          runtime->send_slice_find_intra_space_dependence(orig_proc, rez);
          // Save this is for ourselves
          intra_space_dependences[point] = temp_event;
          return temp_event;
        }
      }
      // If we make it down here then we're on the same node as the 
      // index_owner so we can just as it what the answer and save it
      const RtEvent result = index_owner->find_intra_space_dependence(point);
      AutoLock o_lock(op_lock);
      intra_space_dependences[point] = result;
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_intra_space_dependence(const DomainPoint &point,
                                                  const DomainPoint &next,
                                                  RtEvent point_mapped)
    //--------------------------------------------------------------------------
    {
      // Check to see if we already sent it already
      {
        const std::pair<DomainPoint,DomainPoint> key(point, next);
        AutoLock o_lock(op_lock);
        std::map<DomainPoint,RtEvent>::const_iterator finder = 
          intra_space_dependences.find(point);
        if (finder != intra_space_dependences.end())
        {
#ifdef DEBUG_LEGION
          assert(finder->second == point_mapped);
#endif
          // For control replication we need the index owner to see all
          // the unique sets of dependences, see if we've seen this 
          // combination before, if not, allow it to be sent back
          // to the index owner for it's own visibility
          std::set<std::pair<DomainPoint,DomainPoint> >::const_iterator
            key_finder = unique_intra_space_deps.find(key);
          if (key_finder != unique_intra_space_deps.end())
            return;
        }
        else
          // Otherwise save it and then let it flow back to the index owner
          intra_space_dependences[point] = point_mapped;
        // Always save this if we make it here
        unique_intra_space_deps.insert(key);
      }
      if (is_remote())
      {
        Serializer rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize(point);
          rez.serialize(next);
          rez.serialize(point_mapped);
        }
        runtime->send_slice_record_intra_space_dependence(orig_proc, rez);
      }
      else
        index_owner->record_intra_space_dependence(point, next, point_mapped);
    }

    //--------------------------------------------------------------------------
    size_t SliceTask::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_remote());
#endif
      return points.size();
    }

    //--------------------------------------------------------------------------
    bool SliceTask::find_shard_participants(std::vector<ShardID> &shards)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!is_remote());
#endif
      return index_owner->find_shard_participants(shards);
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::perform_collective_versioning_analysis(unsigned index,
        LogicalRegion handle, EqSetTracker *tracker, const FieldMask &mask,
        unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        return MultiTask::rendezvous_collective_versioning_analysis(index,
            handle, tracker, runtime->address_space, mask, parent_req_index);
      else
        return index_owner->rendezvous_collective_versioning_analysis(index,
            handle, tracker, runtime->address_space, mask, parent_req_index);
    }

    //--------------------------------------------------------------------------
    void SliceTask::perform_replicate_collective_versioning(unsigned index,
        unsigned parent_req_index,
        LegionMap<LogicalRegion,RegionVersioning> &to_perform)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        MultiTask::rendezvous_collective_versioning_analysis(index,
            parent_req_index, to_perform);
      else
        index_owner->rendezvous_collective_versioning_analysis(index,
            parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    void SliceTask::convert_replicate_collective_views(const RendezvousKey &key,
                       std::map<LogicalRegion,CollectiveRendezvous> &rendezvous)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        MultiTask::rendezvous_collective_mapping(key, rendezvous);
      else
        index_owner->rendezvous_collective_mapping(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    void SliceTask::finalize_collective_versioning_analysis(unsigned index,
                          unsigned parent_req_index,
                          LegionMap<LogicalRegion,RegionVersioning> &to_perform)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_remote());
#endif
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(index_owner);
        rez.serialize(index);
        rez.serialize<size_t>(points.size());
        rez.serialize<size_t>(to_perform.size());
        for (LegionMap<LogicalRegion,RegionVersioning>::const_iterator pit =
              to_perform.begin(); pit != to_perform.end(); pit++)
        {
          rez.serialize(pit->first);
#ifdef DEBUG_LEGION
          assert(pit->second.ready_event.exists());
#endif
          rez.serialize(pit->second.ready_event);
          rez.serialize<size_t>(pit->second.trackers.size());
          for (LegionMap<std::pair<AddressSpaceID,EqSetTracker*>,FieldMask>::
                const_iterator it = pit->second.trackers.begin(); it != 
                pit->second.trackers.end(); it++)
          {
            rez.serialize(it->first.first);
            rez.serialize(it->first.second);
            rez.serialize(it->second);
          }
        }
        if (to_perform.empty())
        {
          // If we don't have any local points depending on the result
          // then we need to pack an event to make sure this message gets
          // there before the index task is cleaned up
          const RtUserEvent done_event = Runtime::create_rt_user_event();
          rez.serialize(done_event);
          map_applied_conditions.insert(done_event);  
        }
      }
      runtime->send_slice_remote_versioning_rendezvous(orig_proc, rez);
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::convert_collective_views(unsigned requirement_index,
                       unsigned analysis_index, LogicalRegion region,
                       const InstanceSet &targets, InnerContext *physical_ctx,
                       CollectiveMapping *&analysis_mapping, bool &first_local,
                       LegionVector<FieldMaskSet<InstanceView> > &target_views,
                       std::map<InstanceView*,size_t> &collective_arrivals)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        return MultiTask::convert_collective_views(requirement_index,
            analysis_index, region, targets, physical_ctx, analysis_mapping,
            first_local, target_views, collective_arrivals);
      else
        return index_owner->convert_collective_views(requirement_index,
            analysis_index, region, targets, physical_ctx, analysis_mapping,
            first_local, target_views, collective_arrivals);
    }

    //--------------------------------------------------------------------------
    void SliceTask::rendezvous_collective_mapping(unsigned requirement_index, 
                 unsigned analysis_index, LogicalRegion region,
                 RendezvousResult *result, AddressSpaceID source,
                 const LegionVector<std::pair<DistributedID,FieldMask> > &insts)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(is_remote());
      assert(source == runtime->address_space);
#endif
      // Send this back to the owner node
      Serializer rez;
      {
        RezCheck z(rez);
        rez.serialize(index_owner);
        rez.serialize(requirement_index);
        rez.serialize(analysis_index);
        rez.serialize(region);
        rez.serialize(result);
        rez.serialize<size_t>(insts.size());
        for (LegionVector<std::pair<DistributedID,FieldMask> >::const_iterator
              it = insts.begin(); it != insts.end(); it++)
        {
          rez.serialize(it->first);
          rez.serialize(it->second);
        }
      }
      runtime->send_slice_remote_rendezvous(orig_proc, rez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_collective_rendezvous(
                   Deserializer &derez, Runtime *runtime, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *index_owner;
      derez.deserialize(index_owner);
      unsigned requirement_index, analysis_index;
      derez.deserialize(requirement_index);
      derez.deserialize(analysis_index);
      LogicalRegion region;
      derez.deserialize(region);
      RendezvousResult *result;
      derez.deserialize(result);
      size_t num_insts;
      derez.deserialize(num_insts);
      LegionVector<std::pair<DistributedID,FieldMask> > instances(num_insts);
      for (unsigned idx = 0; idx < num_insts; idx++)
      {
        derez.deserialize(instances[idx].first);
        derez.deserialize(instances[idx].second);
      }

      index_owner->rendezvous_collective_mapping(requirement_index,
          analysis_index, region, result, source, instances);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceTask::handle_collective_versioning_rendezvous(
                                          Deserializer &derez, Runtime *runtime)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask *index_owner;
      derez.deserialize(index_owner);
      unsigned index;
      derez.deserialize(index);
      size_t total_points;
      derez.deserialize(total_points);
      index_owner->unpack_slice_collective_versioning_rendezvous(derez, index, 
                                                                 total_points);
    }
    
  }; // namespace Internal 
}; // namespace Legion 

#undef PRINT_REG

// EOF

