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

#include "legion/tasks/single.h"
#include "legion/contexts/leaf.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/operations/mustepoch.h"
#include "legion/tracing/automatic.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Single Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SingleTask::SingleTask(void) : TaskOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    SingleTask::~SingleTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void SingleTask::activate(void)
    //--------------------------------------------------------------------------
    {
      TaskOp::activate();
      profiling_reported = RtUserEvent::NO_RT_USER_EVENT;
      profiling_priority = LG_THROUGHPUT_WORK_PRIORITY;
      single_task_termination = ApUserEvent::NO_AP_USER_EVENT;
      copy_fill_priority = 0;
      outstanding_profiling_requests.store(0);
      outstanding_profiling_reported.store(0);
      selected_variant = 0;
      task_priority = 0;
      perform_postmap = false;
      execution_context = nullptr;
      remote_trace_recorder = nullptr;
      shard_manager = nullptr;
      leaf_cached = false;
      inner_cached = false;
    }

    //--------------------------------------------------------------------------
    void SingleTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      target_processors.clear();
      physical_instances.clear();
      region_preconditions.clear();
      source_instances.clear();
      future_memories.clear();
      virtual_mapped.clear();
      no_access_regions.clear();
      map_applied_conditions.clear();
      task_profiling_requests.clear();
      copy_profiling_requests.clear();
      untracked_valid_regions.clear();
      if ((execution_context != nullptr) &&
          execution_context->remove_base_gc_ref(SINGLE_TASK_REF))
        delete execution_context;
      if ((shard_manager != nullptr) &&
          shard_manager->remove_base_gc_ref(SINGLE_TASK_REF))
        delete shard_manager;
      for (const std::pair<const std::pair<Memory, bool>, MemoryPool*>&
               pool_pair : leaf_memory_pools)
        delete pool_pair.second;
      leaf_memory_pools.clear();
      legion_assert(remote_trace_recorder == nullptr);
      TaskOp::deactivate(freeop);
    }

    //--------------------------------------------------------------------------
    bool SingleTask::is_leaf(void) const
    //--------------------------------------------------------------------------
    {
      if (!leaf_cached)
      {
        VariantImpl* var =
            runtime->find_variant_impl(task_id, selected_variant);
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
        VariantImpl* var =
            runtime->find_variant_impl(task_id, selected_variant);
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
    void SingleTask::clone_single_from(SingleTask* rhs)
    //--------------------------------------------------------------------------
    {
      this->clone_task_op_from(rhs, this->target_proc, false /*stealable*/);
      this->index_point = rhs->index_point;
      this->virtual_mapped = rhs->virtual_mapped;
      this->no_access_regions = rhs->no_access_regions;
      this->target_processors = rhs->target_processors;
      this->physical_instances = rhs->physical_instances;
      // no need to copy the control replication map
      this->selected_variant = rhs->selected_variant;
      this->task_priority = rhs->task_priority;
      // DON'T CLONE THE SHARD MANAGER! (will be set by the caller)
      // For now don't copy anything else below here
      // In the future we may need to copy the profiling requests
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_single_task(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_base_task(rez, target);
      if (is_origin_mapped())
      {
        rez.serialize(selected_variant);
        rez.serialize(task_priority);
        rez.serialize<size_t>(target_processors.size());
        for (const Processor& processor : target_processors)
          rez.serialize(processor);
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
        {
          // C++ is stupid and tries to convert to a std::optional<bool> here
          rez.serialize<bool>(bool(virtual_mapped[idx]));
          if (virtual_mapped[idx])
            version_infos[idx].pack_equivalence_sets(rez);
        }
        rez.serialize(single_task_termination);
        rez.serialize<size_t>(physical_instances.size());
        for (const InstanceSet& instance : physical_instances)
          instance.pack_references(rez);
        rez.serialize<size_t>(region_preconditions.size());
        for (const ApEvent& event : region_preconditions) rez.serialize(event);
        rez.serialize<size_t>(future_memories.size());
        for (const Memory& memory : future_memories) rez.serialize(memory);
        rez.serialize<size_t>(task_profiling_requests.size());
        for (const ProfilingMeasurementID& request : task_profiling_requests)
          rez.serialize(request);
        rez.serialize<size_t>(copy_profiling_requests.size());
        for (const ProfilingMeasurementID& request : copy_profiling_requests)
          rez.serialize(request);
        if (!task_profiling_requests.empty() ||
            !copy_profiling_requests.empty())
          rez.serialize(profiling_priority);
        rez.serialize<size_t>(untracked_valid_regions.size());
        for (unsigned region_idx : untracked_valid_regions)
          rez.serialize(region_idx);
        rez.serialize<size_t>(leaf_memory_pools.size());
        for (const std::pair<const std::pair<Memory, bool>, MemoryPool*>&
                 pool_pair : leaf_memory_pools)
        {
          rez.serialize(pool_pair.first.first);
          rez.serialize(pool_pair.first.second);
          pool_pair.second->serialize(rez);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::unpack_single_task(
        Deserializer& derez, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
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
            version_infos[idx].unpack_equivalence_sets(derez, ready_events);
        }
        derez.deserialize(single_task_termination);
        size_t num_phy;
        derez.deserialize(num_phy);
        physical_instances.resize(num_phy);
        for (unsigned idx = 0; idx < num_phy; idx++)
          physical_instances[idx].unpack_references(derez, ready_events);
        size_t num_pre;
        derez.deserialize(num_pre);
        region_preconditions.resize(num_pre);
        for (unsigned idx = 0; idx < num_pre; idx++)
          derez.deserialize(region_preconditions[idx]);
        size_t num_future_memories;
        derez.deserialize(num_future_memories);
        future_memories.resize(num_future_memories);
        for (unsigned idx = 0; idx < num_future_memories; idx++)
        {
          derez.deserialize(future_memories[idx]);
          // Safe to block indefinitely here for unbounded pools
          if (future_memories[idx].exists())
            futures[idx].impl->request_application_instance(
                future_memories[idx], this,
                nullptr /*safe_for_unbounded_pools*/);
        }
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
        size_t num_pools;
        derez.deserialize(num_pools);
        for (unsigned idx = 0; idx < num_pools; idx++)
        {
          std::pair<Memory, bool> key;
          derez.deserialize(key.first);
          derez.deserialize(key.second);
          leaf_memory_pools[key] = MemoryPool::deserialize(derez);
        }
      }
      update_no_access_regions();
    }

    //--------------------------------------------------------------------------
    void SingleTask::shard_off(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
      // Still need this to record that this operation is done for LegionSpy
      LegionSpy::log_operation_events(
          unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
      // Do the stuff to record that this is mapped and executed
      complete_mapping(mapped_precondition);
      complete_execution();
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void SingleTask::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
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
            if (is_replicable() && replicate_task())
              return;
            if (perform_mapping())
              launch_task();
          }
        }
        // otherwise it was sent away
      }
      else
      {
        // See if we have a must epoch in which case
        // we can simply record ourselves and we are done
        if (must_epoch == nullptr)
        {
          legion_assert(target_proc.exists());
          // See if this task is going to be sent
          // remotely in which case we need to do the
          // mapping now, otherwise we can defer it
          // until the task ends up on the target processor
          if (is_origin_mapped())
          {
            if (perform_mapping() && distribute_task())
              launch_task();
          }
          else
          {
            if (distribute_task())
            {
              // Still local so try mapping and launching
              if (is_replicable() && replicate_task())
                return;
              if (perform_mapping())
                launch_task();
            }
          }
        }
        else
          must_epoch->register_single_task(this, must_epoch_index);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_inlining(
        VariantImpl* variant, const std::deque<InstanceSet>& parent_instances)
    //--------------------------------------------------------------------------
    {
      legion_assert(parent_instances.size() == regions.size());
      selected_variant = variant->vid;
      target_processors.emplace_back(current_proc);
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
      launch_task(true /*inline*/);
    }

    //--------------------------------------------------------------------------
    void SingleTask::enqueue_ready_task(
        bool use_target_processor, RtEvent wait_on /*=RtEvent::NO_RT_EVENT*/)
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
    RtEvent SingleTask::perform_versioning_analysis(const bool post_mapper)
    //--------------------------------------------------------------------------
    {
      if (is_replaying())
        return RtEvent::NO_RT_EVENT;
      // If we're remote and origin mapped, then we are already done
      if (is_remote() && is_origin_mapped())
        return RtEvent::NO_RT_EVENT;
      legion_assert(
          version_infos.empty() ||
          (version_infos.size() == get_region_count()));
      version_infos.resize(get_region_count());
      std::set<RtEvent> ready_events;
      std::vector<RtEvent> output_events;
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
      {
        if (no_access_regions[idx] || (post_mapper && virtual_mapped[idx]))
          continue;
        VersionInfo& version_info = version_infos[idx];
        if (version_info.has_version_info())
          continue;
        const RegionRequirement& req = logical_regions[idx];
        if ((regions.size() <= idx) && !is_output_valid(idx - regions.size()))
        {
          RtEvent output_ready;
          Operation::perform_versioning_analysis(
              idx, req, version_info, ready_events, &output_ready);
          legion_assert(output_ready.exists());
          output_events.emplace_back(output_ready);
        }
        else
          Operation::perform_versioning_analysis(
              idx, req, version_info, ready_events, nullptr /*output region*/,
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
    void SingleTask::initialize_map_task_input(
        Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
        MustEpochOp* must_epoch_owner)
    //--------------------------------------------------------------------------
    {
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
      if (must_epoch_owner != nullptr)
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
          legion_assert(must_epoch_owner != nullptr);
          // We can skip this since we already know the result
          continue;
        }
        if (request_valid_instances &&
            (regions[idx].privilege != LEGION_REDUCE))
        {
          InstanceSet current_valid;
          local::FieldMaskMap<ReplicatedView> collectives;
          physical_premap_region(
              idx, regions[idx], version_infos[idx], current_valid, collectives,
              map_applied_conditions);
          if (regions[idx].is_no_access())
            prepare_for_mapping(
                current_valid, collectives, input.valid_instances[idx],
                input.valid_collectives[idx]);
          else
            prepare_for_mapping(
                current_valid, collectives, visible_memories,
                input.valid_instances[idx], input.valid_collectives[idx]);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool SingleTask::finalize_map_task_output(
        Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
        MustEpochOp* must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      if (output.abort_mapping)
        return false;
      // At this point we know we're going to map this task here
      // Check to see if we need to make a remote trace recorder
      if (is_remote() && is_recording() && (remote_trace_recorder == nullptr))
      {
        remote_trace_recorder = new RemoteTraceRecorder(
            orig_proc.address_space(), get_trace_local_id(), tpl, 0 /*did*/,
            0 /*tid*/);
        remote_trace_recorder->add_recorder_reference();
        legion_assert(!single_task_termination.exists());
        // Really unusual case here, if we're going to be doing remote tracing
        // then we need to get an event from the owner node because some kinds
        // of tracing (e.g. those with control replication) don't work otherwise
        remote_trace_recorder->request_term_event(single_task_termination);
        record_completion_effect(single_task_termination);
      }
      // Create our task termination event at this point
      // Note that tracing doesn't track this as a user event, it is just
      // a name we're making for the termination event
      if (!single_task_termination.exists())
      {
        single_task_termination = Runtime::create_ap_user_event(nullptr);
        record_completion_effect(single_task_termination);
      }
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      // first check the processors to make sure they are all on the
      // same node and of the same kind, if we know we have a must epoch
      // owner then we also know there is only one valid choice
      if (must_epoch_owner == nullptr)
      {
        if (output.target_procs.empty())
        {
          Warning warning;
          warning << "Empty output target_procs from call to 'map_task' "
                  << "by mapper " << *mapper << " for " << *this
                  << ". Adding the 'target_proc' " << this->target_proc
                  << " as the default.";
          warning.raise();
          output.target_procs.emplace_back(this->target_proc);
        }
        else if (output.target_procs.size() > 1)
        {
          if (concurrent_task)
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Mapper " << *mapper
                << " provided multiple target processors as output "
                << "from 'map_task' for " << *this << " which was launched "
                << "in a concurrent index space task launch. Mappers are only "
                << "permitted to specify a single target processor for mapping "
                << "tasks in concurrent index space task launches.";
            error.raise();
          }
        }
        if (runtime->safe_mapper)
          validate_target_processors(output.target_procs);
        // Save the target processors from the output
        target_processors = output.target_procs;
        target_proc = target_processors.front();
      }
      else
      {
        if (output.target_procs.size() > 1)
        {
          Warning warning;
          warning << "Ignoring spurious additional target processors "
                  << "requested in 'map_task' for " << *this << " by mapper "
                  << *mapper << " because task is part of a must epoch launch.";
          warning.raise();
        }
        if (!output.target_procs.empty() &&
            (output.target_procs[0] != this->target_proc))
        {
          Warning warning;
          warning << "Ignoring processor request of "
                  << output.target_procs.front() << " for " << *this
                  << " by mapper " << *mapper
                  << " because task has already been mapped to processor "
                  << this->target_proc << " as part of a must epoch launch.";
          warning.raise();
        }
        // Only one valid choice in this case, ignore everything else
        target_processors.emplace_back(this->target_proc);
      }
      // If we had any future mapping outputs, we can grab them
      if (!futures.empty())
      {
        future_memories.swap(output.future_locations);
        if (futures.size() != future_memories.size())
          future_memories.resize(futures.size(), Memory::NO_MEMORY);
        // Check to make sure that they are all on the same address
        // space as the target processor(s)
        const AddressSpaceID target_space = this->target_proc.address_space();
        for (unsigned idx = 0; idx < future_memories.size(); idx++)
        {
          if (!future_memories[idx].exists())
            continue;
          if (futures[idx].impl == nullptr)
            continue;
          if (future_memories[idx].address_space() != target_space)
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of 'map_task' on "
                     "mapper "
                  << *mapper << "when mapping " << *this
                  << ". Mapper attempted to map future " << idx << " to memory "
                  << future_memories[idx] << " in address space "
                  << future_memories[idx].address_space()
                  << " which is not the same as address space " << target_space
                  << " of the target processor " << this->target_proc
                  << ". Mapped futures must be in the same "
                  << "address space as the target processor for task mappings.";
            error.raise();
          }
          // Request the future memories be created
          // Safe to block here indefinitely waiting for unbounded pools
          futures[idx].impl->request_application_instance(
              future_memories[idx], this, nullptr /*safe_for_unbounded_pools*/);
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
        for (const ProfilingMeasurementID& measurement_id :
             output.task_prof_requests.requested_measurements)
        {
          if (measurement_id > Mapping::PMID_LEGION_FIRST)
          {
            // If we haven't seen a proc usage yet, then add it
            // to the realm requests to ensure we get a callback
            // for this task. We know we'll see it before this
            // because the measurement IDs are in order
            if (!has_proc_request)
              task_profiling_requests.emplace_back(
                  (ProfilingMeasurementID)Realm::PMID_OP_PROC_USAGE);
            // These are legion profiling requests and currently
            // are only profiling task information
            task_profiling_requests.emplace_back(measurement_id);
            continue;
          }
          switch ((Realm::ProfilingMeasurementID)measurement_id)
          {
            case Realm::PMID_OP_PROC_USAGE:
              has_proc_request = true;  // Then fall through
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
                task_profiling_requests.emplace_back(measurement_id);
                break;
              }
            default:
              {
                Warning warning;
                warning << "Mapper " << *mapper << " requested a profiling "
                        << "measurement of type " << measurement_id
                        << " which is not applicable to " << *this
                        << " and will be ignored.";
                warning.raise();
              }
          }
        }
        legion_assert(!profiling_reported.exists());
        legion_assert(outstanding_profiling_requests == 0);
        profiling_reported = Runtime::create_rt_user_event();
        // Increment the number of profiling responses here since we
        // know that we're going to get one for launching the task
        // No need for the lock since no outstanding physical analyses
        // can be running yet
        outstanding_profiling_requests = 1;
      }
      if (!output.copy_prof_requests.empty())
      {
        filter_copy_request_kinds(
            mapper, output.copy_prof_requests.requested_measurements,
            copy_profiling_requests, true /*warn*/);
        profiling_priority = output.profiling_priority;
        if (!profiling_reported.exists())
          profiling_reported = Runtime::create_rt_user_event();
      }
      // See whether the mapper picked a variant or a generator
      VariantImpl* variant_impl = nullptr;
      if (output.chosen_variant > 0)
        variant_impl = runtime->find_variant_impl(
            task_id, output.chosen_variant, true /*can fail*/);
      else  // TODO: invoke a generator if one exists
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error
            << "Invalid mapper output from invocation of 'map_task' on mapper "
            << *mapper << ". Mapper specified an invalid task variant "
            << "of ID 0 for " << *this << ", but Legion does not yet "
            << "support task generators.";
        error.raise();
      }
      if (variant_impl == nullptr)
      {
        // If we couldn't find or make a variant that is bad
        Error error(LEGION_MAPPER_EXCEPTION);
        error
            << "Invalid mapper output from invocation of 'map_task' on mapper "
            << *mapper << ". Mapper failed to specify a valid "
            << "task variant or generator capable of create a variant "
            << "implementation of " << *this << ".";
        error.raise();
      }
      // Record the future output size
      if (!elide_future_return)
      {
        if (future_return_size)
          handle_future_size(*future_return_size, map_applied_conditions);
        else if (variant_impl->has_return_type_size)
        {
          future_return_size = variant_impl->return_type_size;
          handle_future_size(*future_return_size, map_applied_conditions);
        }
      }
      // Create ny memory pools if this is a leaf task variant
      // Note this has to come AFTER we create the future instances or we
      // could accidentally end up blocking ourselves from doing memory
      // allocations if we have an unbounded pool
      if (variant_impl->is_leaf())
      {
        std::map<std::pair<Memory, bool>, PoolBounds> dynamic_pool_bounds;
        // Combine the escaping and non-escaping data structures together
        // Should really have done this from the start but didn't think
        // the pool design through completely
        for (const std::pair<const Memory, PoolBounds>& bounds :
             output.leaf_pool_bounds)
          dynamic_pool_bounds.emplace(std::make_pair(
              std::make_pair(bounds.first, true /*escaping*/), bounds.second));
        for (const std::pair<const Memory, PoolBounds>& bounds :
             output.non_escaping_leaf_pool_bounds)
        {
          if (!bounds.second.is_bounded())
          {
            Warning warning;
            warning << "Detected request for non-escaping unbounded pool "
                    << " in memory " << bounds.first << " by 'map_task' for "
                    << *this << " by mapper " << *mapper << ". ";
            if (dynamic_pool_bounds
                    .emplace(std::make_pair(
                        std::make_pair(bounds.first, true /*escaping*/),
                        bounds.second))
                    .second)
              warning
                  << "Promoted this request up to an escaping unbounded pool.";
            else
              warning << "Another kind of escaping pool was already requested "
                      << "for this memory so this request will be ignored.";
            warning.raise();
          }
          else
            dynamic_pool_bounds.emplace(std::make_pair(
                std::make_pair(bounds.first, false /*escaping*/),
                bounds.second));
        }
        create_leaf_memory_pools(variant_impl, dynamic_pool_bounds);
      }
      else
      {
        // If we're a concurrent task or a collectively mapped task then we
        // still need to participate in the max-allreduce for allocating
        // unbounded pools
        if (concurrent_task || must_epoch_task ||
            !check_collective_regions.empty())
          order_collectively_mapped_unbounded_pools(0, false /*need result*/);
        if (!leaf_memory_pools.empty())
        {
          // Free up any leaf memory pools that we have since we don't need them
          for (const std::pair<const std::pair<Memory, bool>, MemoryPool*>&
                   pool_pair : leaf_memory_pools)
            delete pool_pair.second;
          leaf_memory_pools.clear();
        }
      }
      // Save variant validation until we know which instances we'll be using
      // fill in virtual_mapped
      virtual_mapped.resize(logical_regions.size(), false);
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
      if (runtime->safe_mapper)
      {
        if (target_processors.size() > 1)
        {
          // If we have multiple processor, we want the set of
          // memories visible to all of them
          Machine::MemoryQuery visible_query(runtime->machine);
          for (const Processor& processor : target_processors)
            visible_query.has_affinity_to(processor);
          for (const Memory& memory : visible_query)
            visible_memories.insert(memory);
        }
        else
          runtime->find_visible_memories(target_proc, visible_memories);
      }
      if (this->must_epoch != nullptr)
      {
        // Merge the must epoch owners acquired instances too
        // if we need to check for all our instances being acquired
        const std::map<PhysicalManager*, unsigned>* epoch_acquired =
            this->must_epoch->get_acquired_instances_ref();
        if (epoch_acquired != nullptr)
        {
          for (const std::pair<PhysicalManager* const, unsigned>& manager_pair :
               *epoch_acquired)
          {
            if (acquired_instances.find(manager_pair.first) !=
                acquired_instances.end())
              continue;
            // Can safely add another reference since we know one is
            // already being held by the must-epoch operation
            manager_pair.first->add_base_valid_ref(MAPPING_ACQUIRE_REF);
            acquired_instances.emplace(std::make_pair(manager_pair.first, 1));
          }
        }
      }
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        // Skip any NO_ACCESS or empty privilege field regions
        if (no_access_regions[idx])
          continue;
        // Do the conversion
        InstanceSet& result = physical_instances[idx];
        RegionTreeID bad_tree = 0;
        std::vector<FieldID> missing_fields;
        std::vector<PhysicalManager*> unacquired;
        // Convert any sources first
        if (!output.source_instances[idx].empty())
          physical_convert_sources(
              regions[idx], output.source_instances[idx], source_instances[idx],
              &acquired_instances);
        int composite_idx = physical_convert_mapping(
            regions[idx], output.chosen_instances[idx], result, bad_tree,
            missing_fields, &acquired_instances, unacquired,
            runtime->safe_mapper);
        if (bad_tree > 0)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'map_task' on "
                   "mappper "
                << *mapper << ". Mapper specified an instance from region tree "
                << bad_tree << " for use with region requirement " << idx
                << " of " << *this << " whose region is from region tree "
                << regions[idx].region.get_tree_id() << ".";
          error.raise();
        }
        if (!missing_fields.empty())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'map_task' on "
                   "mapper "
                << *mapper << ". Mapper failed to specify an instance for "
                << missing_fields.size() << " fields of region requirement "
                << idx << " of " << *this << ". The missing fields are: ";
          bool first = true;
          for (const FieldID& field_id : missing_fields)
          {
            const void* name;
            size_t name_size;
            if (!runtime->retrieve_semantic_information(
                    regions[idx].region.get_field_space(), field_id,
                    LEGION_NAME_SEMANTIC_TAG, name, name_size,
                    true /*can fail*/, false))
              name = "(no name)";
            if (first)
              first = false;
            else
              error << ", ";
            error << static_cast<const char*>(name) << " (FieldID: " << field_id
                  << ")";
          }
        }
        if (!unacquired.empty())
        {
          for (PhysicalManager* const manager : unacquired)
          {
            if (acquired_instances.find(manager) == acquired_instances.end())
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error
                  << "Invalid mapper output from 'map_task' invocation on "
                     "mapper "
                  << *mapper
                  << ". Mapper selected physical instance for region "
                  << "requirement " << idx << " of " << *this
                  << " which has already "
                  << "been collected. If the mapper had properly acquired this "
                  << "instance as part of the mapper call it would have "
                     "detected "
                  << "this. Please update the mapper to abide by proper "
                     "mapping "
                  << "conventions.";
              error.raise();
            }
          }
          // Event if we did successfully acquire them, still issue the warning
          Warning warning;
          warning << "Mapper " << *mapper << " failed to acquire instances "
                  << "for region requirement " << idx << " of " << *this
                  << "in 'map_task' call. You may experience undefined "
                  << "behavior as a consequence.";
          warning.raise();
        }
        // See if they want a virtual mapping
        if (composite_idx >= 0)
        {
          // Everything better be all virtual or all real
          if (result.size() > 1)
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of 'map_task' on "
                     "mapper "
                  << *mapper << ". Mapper specified mixed virtual and concrete "
                  << "instances for region requirement " << idx << " of "
                  << *this
                  << ". Only full concrete instances or a single virtual "
                     "instance "
                  << "is supported.";
            error.raise();
          }
          if (!IS_EXCLUSIVE(regions[idx]))
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'map_task' on "
                   "mapper "
                << *mapper
                << ". Illegal composite instance requested on region "
                << "requirement " << idx << " of " << *this
                << " which has a relaxed coherence mode. Virtual mappings are "
                << "only permitted for exclusive coherence.";
            error.raise();
          }
          virtual_mapped[idx] = true;
        }
        // Skip checks if the mapper promises it is safe
        if (!runtime->safe_mapper)
          continue;
        // If this is anything other than a virtual mapping, check that
        // the instances align with the privileges
        if (!virtual_mapped[idx])
        {
          std::vector<LogicalRegion> regions_to_check(1, regions[idx].region);
          for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
          {
            PhysicalManager* manager = result[idx2].get_physical_manager();
            if (!manager->meets_regions(regions_to_check))
            {
              // Doesn't satisfy the region requirement
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Invalid mapper output from invocation of 'map_task' on "
                       "mapper "
                    << *mapper
                    << ". Mapper specified instance that does not meet "
                    << "region requirement " << idx << " for " << *this
                    << ". The index space for the instance has insufficient "
                       "space "
                    << "for the requested logical region.";
              error.raise();
            }
          }
          if (!regions[idx].is_no_access() &&
              !variant_impl->is_no_access_region(idx))
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
            {
              const Memory mem = result[idx2].get_memory();
              if (visible_memories.find(mem) == visible_memories.end())
              {
                // Not visible from all target processors
                Error error(LEGION_MAPPER_EXCEPTION);
                error
                    << "Invalid mapper output from invocation of 'map_task' on "
                       "mapper "
                    << *mapper << ". Mapper selected an instance for region "
                    << "requirement " << idx << " in memory " << mem
                    << " which is not visible from the target processors for "
                    << *this << ".";
                error.raise();
              }
            }
          }
          // If this is a reduction region requirement make sure all the
          // managers are reduction instances with the right reduction ops
          if (IS_REDUCE(regions[idx]))
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
            {
              PhysicalManager* manager = result[idx2].get_physical_manager();
              if (!manager->is_reduction_manager())
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error
                    << "Invalid mapper output from invocation of 'map_task' "
                    << "on mapper " << *mapper << ". Mapper failed to choose a "
                    << "specialized reduction instance for region requirement "
                    << idx << " of " << *this
                    << " which has reduction privileges.";
                error.raise();
              }
              else if (manager->redop != regions[idx].redop)
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error
                    << "Invalid mapper output from invocation of 'map_task' on "
                       "mapper "
                    << *mapper
                    << ". Mapper failed selected a specialized reduction "
                    << "instance with reduction operator " << manager->redop
                    << " for region requirement " << idx << " of " << *this
                    << " which has reduction privileges on a different "
                       "reduction "
                    << "operator " << regions[idx].redop << ".";
                error.raise();
              }
            }
          }
          else
          {
            for (unsigned idx2 = 0; idx2 < result.size(); idx2++)
              if (result[idx2].get_manager()->is_reduction_manager())
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error
                    << "Invalid mapper output from invocation of 'map_task' on "
                       "mapper "
                    << *mapper
                    << ". Mapper selected illegal specialized reduction "
                    << "instance for region requirement " << idx << " of "
                    << *this << " which does not have reduction privileges.";
                error.raise();
              }
          }
        }
      }
      // This is a bit of a hairy case: since leaf tasks do not hold valid
      // references on their instances after mapping, the mapped instances
      // can become eligible for deferred deletions. However, if we have an
      // unbounded pool then it can still try to do deferred allocations
      // which might try to collect the instances we mapped in this task.
      // To prevent this unbound pools need to capture valid references on
      // any acquired instances for this task to ensure that we don't try
      // to use our own instances to satisfy an unbounded pool allocation.
      for (const std::pair<const std::pair<Memory, bool>, MemoryPool*>&
               pool_pair : leaf_memory_pools)
        pool_pair.second->capture_task_instances(acquired_instances);
      if (!output_regions.empty())
      {
        // Now we prepare output instances
        if (runtime->safe_mapper)
        {
          for (unsigned idx = 0; idx < output_regions.size(); idx++)
          {
            Memory target = output.output_targets[idx];
            if (!target.exists() ||
                visible_memories.find(target) == visible_memories.end())
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Invalid mapper output from invocation of 'map_task' on "
                       "mapper "
                    << *mapper << ". Mapper selected invalid target memory "
                    << target << " for output region requirement " << idx
                    << " of " << *this << ".";
              error.raise();
            }
          }
        }
        const size_t output_offset = regions.size();
        for (unsigned idx = 0; idx < output_regions.size(); idx++)
          prepare_output_instance(
              idx, physical_instances[output_offset + idx], output_regions[idx],
              output.output_targets[idx], output.output_constraints[idx]);
      }
      // If the variant has padded fields we need to get the atomic locks
      if (variant_impl->needs_padding)
        variant_impl->find_padded_locks(this, regions, physical_instances);
      // Now that we have our physical instances we can validate the variant
      if (runtime->safe_mapper)
      {
        legion_assert(!target_processors.empty());
        validate_variant_selection(
            mapper, variant_impl, target_processors.front().kind(),
            physical_instances, "map_task");
      }
      // Record anything else that needs to be recorded
      selected_variant = output.chosen_variant;
      task_priority = output.task_priority;
      perform_postmap = output.postmap_task;
      if (!output.untracked_valid_regions.empty())
      {
        for (const unsigned& region_idx : output.untracked_valid_regions)
        {
          // Remove it if it is too big or is not read-only
          if ((region_idx >= regions.size()) ||
              !IS_READ_ONLY(regions[region_idx]))
          {
            if (region_idx < regions.size())
            {
              Warning warning;
              warning << "Ignoring request by mapper " << *mapper
                      << " to not track valid instances for region requirement "
                      << region_idx << " of " << *this
                      << " because region requirement "
                      << "does not have read-only privileges.";
              warning.raise();
            }
          }
          else
            untracked_valid_regions.emplace_back(region_idx);
        }
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void SingleTask::prepare_output_instance(
        unsigned index, InstanceSet& instance_set, const RegionRequirement& req,
        Memory target, const LayoutConstraintSet& c)
    //--------------------------------------------------------------------------
    {
      MemoryManager* memory_manager = runtime->find_memory_manager(target);

      LayoutConstraintSet constraints;
      constraints.add_constraint(MemoryConstraint(target.kind()))
          .add_constraint(
              SpecializedConstraint(LEGION_AFFINE_SPECIALIZE, 0, false, true))
          .add_constraint(c.ordering_constraint);

      const std::vector<DimensionKind>& ordering =
          constraints.ordering_constraint.ordering;
      if (ordering.empty())
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error
            << "An ordering constraint must be specified for each output "
            << "region, but the mapper did not specify any ordering constraint "
            << "for output region " << index << " of " << *this << ".";
        error.raise();
      }
      else if (static_cast<int>(ordering.size()) != req.region.get_dim() + 1)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "The mapper chose an ordering constraint with "
              << (ordering.size() - 1) << " dimensions for output region "
              << index << " of " << *this << ", but the region has "
              << req.region.get_dim()
              << " dimensions. Make sure you specify a correct ordering.";
        error.raise();
      }
      // TODO: For now we only allow SOA layout with either the C order
      // or the Fotran order for output instances.
      // We've actually added support for this in the OutputRegionImpl
      // but it's unclear what the right way to expose it is since we
      // need to know how to make managers for which groups of fields.
      // Right now the OutputRegionImpl just keys off the grouped_fields
      // parameter which assumes that AOS and hybrid have to be grouped
      // whereas SOA can do whatever it wants and the right thing will happen
      // We assume that if you have grouped fields then there is exactly
      // one PhysicalManager for all the fields, but if not then we will
      // break things up so there is one PhysicalManager per field, it's
      // unclear if we also want to handle the case where there are subsets
      // of fields that share a manager.
      else if (ordering.back() != LEGION_DIM_F)
      {
        Fatal fatal;
        fatal << "Legion currently supports only the SOA layout for output "
                 "regions, "
              << "but output region " << index << " of " << *this
              << " is mapped to a "
              << "non-SOA layout. Please update the mapper to use SOA layout "
              << "for all output regions.";
        fatal.raise();
      }
      std::map<FieldID, std::pair<EqualityKind, size_t> > alignments;
      std::map<FieldID, off_t> offsets;

      for (const AlignmentConstraint& constraint : c.alignment_constraints)
      {
        legion_assert(alignments.find(constraint.fid) == alignments.end());
        alignments[constraint.fid] =
            std::make_pair(constraint.eqk, constraint.alignment);
      }

      for (const OffsetConstraint& constraint : c.offset_constraints)
      {
        legion_assert(offsets.find(constraint.fid) == offsets.end());
        offsets[constraint.fid] = constraint.offset;
      }

      for (const FieldID& fid : req.privilege_fields)
      {
        // Create a layout description with a single field
        std::vector<FieldID> fields(1, fid);
        constraints.field_constraint = FieldConstraint(fields, false, false);

        {
          std::map<FieldID, std::pair<EqualityKind, size_t> >::iterator finder =
              alignments.find(fid);
          if (finder != alignments.end())
            constraints.add_constraint(AlignmentConstraint(
                finder->first, finder->second.first, finder->second.second));
        }

        {
          std::map<FieldID, off_t>::iterator finder = offsets.find(fid);
          if (finder != offsets.end())
            constraints.add_constraint(
                OffsetConstraint(finder->first, finder->second));
        }

        legion_assert(single_task_termination.exists());

        // Create a physical manager that is not bound to any instance
        PhysicalManager* manager = memory_manager->create_unbound_instance(
            req.region, constraints, single_task_termination, 0 /*priority*/);

        // Add an instance ref of the new manager to the instance set
        instance_set.add_instance(
            InstanceRef(manager, manager->layout->allocated_fields));

        // Add the manager to the map of acquired instances so that
        // later we can release it properly
        acquired_instances.insert(std::make_pair(manager, 1));

        constraints.alignment_constraints.clear();
        constraints.offset_constraints.clear();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_replay_operation(unique_op_id);
      std::map<std::pair<Memory, bool>, PoolBounds> pool_bounds;
      tpl->get_mapper_output(
          this, selected_variant, task_priority, perform_postmap,
          target_processors, future_memories, pool_bounds, physical_instances);
      // Then request any future mappings in advance
      if (!futures.empty())
      {
        for (unsigned idx = 0; idx < futures.size(); idx++)
        {
          if (futures[idx].impl == nullptr)
            continue;
          const Memory memory = future_memories[idx];
          if (memory.exists())
            // Safe to block here indefinitely waiting for unbounded pools
            futures[idx].impl->request_application_instance(
                memory, this, nullptr /*safe_for_unbounded_pools*/);
        }
      }
      if (!single_task_termination.exists())
      {
        single_task_termination = Runtime::create_ap_user_event(nullptr);
        record_completion_effect(single_task_termination);
      }
      RtEvent safe_single_task_termination;
      // Make any memory pools required to replay this task
      for (const std::pair<const std::pair<Memory, bool>, PoolBounds>&
               pool_pair : pool_bounds)
      {
        MemoryManager* manager =
            runtime->find_memory_manager(pool_pair.first.first);
        // Recompute these each time as they might be consumed each time
        TaskTreeCoordinates coordinates;
        compute_task_tree_coordinates(coordinates);
        // Safe to block here indefinitely for unbounded pools since any
        // unbounded pools have to come from before the trace
        MemoryPool* pool = manager->create_memory_pool(
            get_unique_id(), coordinates, pool_pair.second,
            nullptr /*safe_for_unbounded_pools*/);
        if (pool == nullptr)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to reserve a dynamic memory pool of "
              << pool_pair.second.size << " bytes for " << *this << " in "
              << manager->get_name()
              << " memory during trace replay. You are actually out of memory "
              << "here so you'll need to either allocate more memory for this "
              << "kind of memory when you configure Realm which may "
                 "necessitate "
              << "finding a bigger machine.";
          error.raise();
        }
        // Finalize any non-escaping pools
        if (!pool_pair.first.second)
        {
          if (!safe_single_task_termination.exists())
          {
            legion_assert(single_task_termination.exists());
            safe_single_task_termination =
                Runtime::protect_event(single_task_termination);
          }
          pool->finalize_pool(safe_single_task_termination);
        }
        leaf_memory_pools.emplace(std::make_pair(pool_pair.first, pool));
      }
      // Make sure to propagate any future sizes that we know about here
      if (!elide_future_return)
      {
        if (future_return_size)
          handle_future_size(*future_return_size, map_applied_conditions);
        else
        {
          VariantImpl* variant_impl =
              runtime->find_variant_impl(task_id, selected_variant);
          legion_assert(variant_impl->has_return_type_size);
          future_return_size = variant_impl->return_type_size;
          // Record the future output size
          handle_future_size(*future_return_size, map_applied_conditions);
        }
      }
      set_origin_mapped(true);  // it's like this was origin mapped
      // should only be replaying leaf tasks currently
      // until we figure out how to handle non-leaf tasks
      legion_assert(is_leaf());
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
      legion_assert(output_regions.empty());
      legion_assert(single_task_termination.exists());
      virtual_mapped.resize(regions.size(), false);
      bool needs_reservations = false;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        InstanceSet& instances = physical_instances[idx];
        if (IS_NO_ACCESS(regions[idx]))
          continue;
        if (IS_ATOMIC(regions[idx]) || IS_REDUCE(regions[idx]))
          needs_reservations = true;
        if (instances.is_virtual_mapping())
          virtual_mapped[idx] = true;
      }
      if (needs_reservations)
        // We group all reservations together anyway
        tpl->get_task_reservations(this, atomic_locks);
      return single_task_termination;
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_replicate_collective_versioning(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, CollectiveVersioningBase::RegionVersioning>&
            to_perform)
    //--------------------------------------------------------------------------
    {
      legion_assert(shard_manager != nullptr);
      legion_assert(!IS_COLLECTIVE(regions[index]));
      legion_assert(!std::binary_search(
          check_collective_regions.begin(), check_collective_regions.end(),
          index));
      // Bounce it back onto the shard manager to finalize
      shard_manager->finalize_replicate_collective_versioning(
          index, parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    void SingleTask::convert_replicate_collective_views(
        const CollectiveViewCreatorBase::RendezvousKey& key,
        std::map<
            LogicalRegion, CollectiveViewCreatorBase::CollectiveRendezvous>&
            rendezvous)
    //--------------------------------------------------------------------------
    {
      legion_assert(shard_manager != nullptr);
      legion_assert(!IS_COLLECTIVE(regions[key.region_index]));
      legion_assert(!std::binary_search(
          check_collective_regions.begin(), check_collective_regions.end(),
          key.region_index));
      shard_manager->finalize_replicate_collective_views(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    InnerContext* SingleTask::create_implicit_context(void)
    //--------------------------------------------------------------------------
    {
      // Make sure we have an implicit profiler for the mapper call
      if ((runtime->profiler != nullptr) && (implicit_profiler == nullptr))
        runtime->profiler->instantiate_profiling_instance();
      legion_assert(output_regions.empty());
      Mapper::ContextConfigOutput configuration;
      configure_execution_context(configuration);

      InnerContext* inner_ctx = nullptr;
      if (configuration.auto_tracing_enabled)
      {
        log_auto_trace.info(
            "Initializing auto tracing for %s (UID %lld)", get_task_name(),
            get_unique_id());
        inner_ctx = new AutoTracing<InnerContext>(
            configuration, this, get_depth(), false /*is inner*/, regions,
            output_regions, parent_req_indexes, virtual_mapped, task_priority,
            ApEvent::NO_AP_EVENT, 0 /*did*/, false /*inline*/,
            true /*implicit*/);
      }
      else
        inner_ctx = new InnerContext(
            configuration, this, get_depth(), false /*is inner*/, regions,
            output_regions, parent_req_indexes, virtual_mapped, task_priority,
            ApEvent::NO_AP_EVENT, 0 /*did*/, false /*inline*/,
            true /*implicit*/);
      execution_context = inner_ctx;
      execution_context->add_base_gc_ref(SINGLE_TASK_REF);
      // Implicit don't have variants so we have to set these properties
      // here, implicit top-level tasks are never leaf or inner tasks
      is_leaf_result = false;
      leaf_cached = true;
      is_inner_result = false;
      inner_cached = true;
      return inner_ctx;
    }

    //--------------------------------------------------------------------------
    void SingleTask::configure_execution_context(
        Mapper::ContextConfigOutput& context_configuration)
    //--------------------------------------------------------------------------
    {
      context_configuration.max_window_size = runtime->initial_task_window_size;
      context_configuration.hysteresis_percentage =
          runtime->initial_task_window_hysteresis;
      context_configuration.max_outstanding_frames = 0;
      context_configuration.min_tasks_to_schedule =
          runtime->initial_tasks_to_schedule;
      context_configuration.min_frames_to_schedule = 0;
      context_configuration.meta_task_vector_width =
          runtime->initial_meta_task_vector_width;
      context_configuration.auto_tracing_enabled = !runtime->no_auto_tracing;
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_configure_context(this, context_configuration);
      // Do a little bit of checking on the output.  Make
      // sure that we only set one of the two cases so we
      // are counting by frames or by outstanding tasks.
      if ((context_configuration.min_tasks_to_schedule == 0) &&
          (context_configuration.min_frames_to_schedule == 0))
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from call 'configure_context' "
              << "on mapper " << *mapper
              << ". One of 'min_tasks_to_schedule' and "
              << "'min_frames_to_schedule' must be non-zero for " << *this
              << ".";
        error.raise();
      }
      // Hysteresis percentage is an unsigned so can't be less than 0
      if (context_configuration.hysteresis_percentage > 100)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from call 'configure_context' "
              << "on mapper " << *mapper << ". The 'hysteresis_percentage' "
              << context_configuration.hysteresis_percentage << " is not "
              << "a value between 0 and 100 for " << *this << ".";
        error.raise();
      }
      if (context_configuration.meta_task_vector_width == 0)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from call 'configure context' "
              << "on mapper " << *this << " for " << *this << ". The "
              << "'meta_task_vector_width' must be a non-zero value.";
        error.raise();
      }
      if (context_configuration.max_templates_per_trace == 0)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from call 'configure context' "
              << "on mapper " << *mapper << " for " << *this << ". The "
              << "'max_templates_per_trace' must be a non-zero value.";
        error.raise();
      }
      if (context_configuration.auto_tracing_enabled && runtime->no_tracing)
      {
        Warning warning;
        warning << "Waring disabling automatic tracing requested by mapper "
                << *mapper << " for " << *this << " because tracing was "
                << "disabled on the command line.";
        warning.raise();
        context_configuration.auto_tracing_enabled = false;
      }
      // If we're counting by frames set min_tasks_to_schedule to zero
      if (context_configuration.min_frames_to_schedule > 0)
        context_configuration.min_tasks_to_schedule = 0;
      // otherwise we know min_frames_to_schedule is zero
    }

    //--------------------------------------------------------------------------
    void SingleTask::set_shard_manager(ShardManager* manager)
    //--------------------------------------------------------------------------
    {
      legion_assert(shard_manager == nullptr);
      shard_manager = manager;
      shard_manager->add_base_gc_ref(SINGLE_TASK_REF);
    }

    //--------------------------------------------------------------------------
    void SingleTask::validate_target_processors(
        const std::vector<Processor>& processors) const
    //--------------------------------------------------------------------------
    {
      legion_assert(!processors.empty());
      // Make sure that they are all on the same node and of the same kind
      const Processor& first = processors.front();
      const Processor::Kind kind = first.kind();
      if (is_concurrent() &&
          ((kind == Processor::PROC_GROUP) || (kind == Processor::PROC_SET)))
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output. Mapper " << *mapper
              << "requested a group processor for " << *this
              << " in a concurrent index space task launch. When mapping "
              << "a concurrent index space task launch then group "
              << "processors are not permitted.";
        error.raise();
      }
      const AddressSpace space = first.address_space();
      for (unsigned idx = 0; idx < processors.size(); idx++)
      {
        const Processor& proc = processors[idx];
        if (!proc.exists())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output. Mapper " << *mapper
                << " requested an "
                << "illegal NO_PROC for a target processor when mapping "
                << *this << ".";
          error.raise();
        }
        else if (proc.kind() != kind)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output. Mapper " << *mapper << " requested "
                << proc.kind() << " processor " << proc << " when mapping "
                << *this << ", but the target processor " << this->target_proc
                << " is a " << this->target_proc.kind()
                << "processor. Only one kind of processor is permitted.";
          error.raise();
        }
        if (proc.address_space() != space)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output. Mapper " << *mapper
                << " requested processor " << proc
                << " which is in address space " << proc.address_space()
                << " when mapping " << *this << " but the target processor "
                << this->target_proc << " is in address space " << space
                << ". All target processors must be in the same address space.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    bool SingleTask::invoke_mapper(MustEpochOp* must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      Mapper::MapTaskInput input;
      Mapper::MapTaskOutput output;
      // Initialize the mapping input which also does all the traversal
      // down to the target nodes
      initialize_map_task_input(input, output, must_epoch_owner);
      // Now we can invoke the mapper to do the mapping
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_map_task(this, input, output);
      // Now we can convert the mapper output into our physical instances
      if (!finalize_map_task_output(input, output, must_epoch_owner))
      {
        // Put this back on the ready queue to map again later
        enqueue_ready_task(false /*target*/);
        return false;
      }
      copy_fill_priority = output.copy_fill_priority;
      if (is_recording())
      {
        legion_assert(
            (remote_trace_recorder != nullptr) ||
            ((tpl != nullptr) && tpl->is_recording()));
        legion_assert(futures.size() == future_memories.size());
        // We swapped this in finalize output so we need to restore it
        // here if we're going to record it
        if (!futures.empty())
          output.future_locations = future_memories;
        // Make sure we save all the future pool bounds sizes including
        // the ones that come statically from the task variant
        for (const std::pair<const std::pair<Memory, bool>, MemoryPool*>&
                 pool_pair : leaf_memory_pools)
        {
          if (pool_pair.first.second)
          {
            // Escaping pool case
            std::map<Memory, PoolBounds>::const_iterator finder =
                output.leaf_pool_bounds.find(pool_pair.first.first);
            if (finder == output.leaf_pool_bounds.end())
              finder = output.leaf_pool_bounds
                           .emplace(std::make_pair(
                               pool_pair.first.first,
                               pool_pair.second->get_bounds()))
                           .first;
            // Issue a warning to the user if the pool is unbounded that
            // this is going to invalidate the trace capture
            if (!finder->second.is_bounded())
            {
              MemoryManager* manager =
                  runtime->find_memory_manager(pool_pair.first.first);
              Warning warning;
              warning
                  << "Detected unbounded pool in trace. Mapper " << *mapper
                  << " requested to trace task " << *this
                  << " with an unbounded memory pool in " << manager->get_name()
                  << " memory " << manager->memory
                  << ". Unbounded pools are not permitted in traces and will "
                  << "prevent this recording of the trace from being replayed.";
              warning.raise();
            }
          }
          else
          {
            // non-escaping pool case
            std::map<Memory, PoolBounds>::const_iterator finder =
                output.non_escaping_leaf_pool_bounds.find(
                    pool_pair.first.first);
            if (finder == output.non_escaping_leaf_pool_bounds.end())
              output.non_escaping_leaf_pool_bounds.emplace(std::make_pair(
                  pool_pair.first.first, pool_pair.second->get_bounds()));
          }
        }
        const TraceLocalID tlid = get_trace_local_id();
        VariantImpl* variant_impl = runtime->find_variant_impl(
            task_id, output.chosen_variant, false /*can fail*/);
        if (remote_trace_recorder != nullptr)
          remote_trace_recorder->record_mapper_output(
              tlid, output, physical_instances, variant_impl->is_leaf(),
              variant_impl->has_return_type_size, map_applied_conditions);
        else
          tpl->record_mapper_output(
              tlid, output, physical_instances, variant_impl->is_leaf(),
              variant_impl->has_return_type_size, map_applied_conditions);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::replicate_task(void)
    //--------------------------------------------------------------------------
    {
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      // There are some local invariants checked here, but there are more
      // of them checked in select_task_options right after the mapper call
      // that decides whether we're going to try to replicate this task
      if (is_recording())
      {
        Warning warning;
        warning
            << "Unsupported request to replicate " << *this << " during "
            << "trace capture by " << *mapper
            << ". Legion does not currently support "
            << "replication of tasks inside of physical traces at the moment. "
            << "You can request support for this feature by emailing the "
            << "the Legion developers list or opening a github issue. The "
            << "mapper call to replicate_task is being elided.";
        warning.raise();
        replicate = false;
        return false;
      }
      Mapper::ReplicateTaskInput input;
      Mapper::ReplicateTaskOutput output;
      mapper->invoke_replicate_task(this, input, output);
      // If we don't have more than one target processor then we're not
      // actually going to replicate this task
      if (output.target_processors.empty())
      {
        replicate = false;
        return false;
      }
      else if (output.target_processors.size() == 1)
      {
        Warning warning;
        warning
            << "Mapper " << *mapper << " requested to replicate " << *this
            << " but only reqeuested one shard to be made. Since one shard "
               "does "
            << "not actually constitute replication, Legion is ignoring this "
            << "request and the task will be mapped like normal. If the "
            << "mapper intended to not perform replication it should return "
            << "an empty vector of target processors for 'replicate_task'.";
        warning.raise();
        replicate = false;
        return false;
      }
      VariantImpl* var_impl = nullptr;
      if (output.leaf_variants.empty())
      {
        var_impl = runtime->find_variant_impl(
            task_id, output.chosen_variant, true /*can_fail*/);
        if (var_impl == nullptr)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'replicate_task' "
                << "on mapper " << *mapper
                << ". Mapper selected an invalid task "
                << "variant " << output.chosen_variant << " for " << *this
                << " that was chosen to be replicated.";
          error.raise();
        }
        // Check that the chosen variant is replicable
        if (!var_impl->is_replicable())
        {
          Warning warning;
          warning
              << "Invalid mapper output from invocation of 'replicate_task' "
              << "on mapper " << *mapper
              << ". Mapper failed to pick an valid task variant "
              << output.chosen_variant << " for " << *this
              << "that was chosen to be replicated. Task variants selected for "
              << "replication must be marked as replicable variants.";
          warning.raise();
        }
      }
      else
      {
        output.chosen_variant = 0;
        if (output.leaf_variants.size() != output.target_processors.size())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'replicate_task' "
                << "on mapper " << *mapper
                << ". Mapper provided %zd leaf variants for "
                << output.leaf_variants.size() << " target processors for "
                << *this
                << ". The same number of leaf variants must be provided "
                << "as target processors.";
          error.raise();
        }
        for (unsigned idx = 0; idx < output.leaf_variants.size(); idx++)
        {
          VariantImpl* impl = runtime->find_variant_impl(
              task_id, output.leaf_variants[idx], true /*can_fail*/);
          if (impl == nullptr)
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'replicate_task' "
                << "on mapper " << *mapper
                << ". Mapper selected an invalid leaf task variant "
                << output.leaf_variants[idx] << " for " << *this
                << " that was chosen to be replicated.";
            error.raise();
          }
          if (var_impl == nullptr)
            var_impl = impl;
          // Check that the chosen variant is a leaf
          if (!impl->is_leaf())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'replicate_task' "
                   "on mapper "
                << *mapper << ". Mapper failed to pick an valid task variant "
                << output.leaf_variants[idx] << " for " << *this
                << " that was chosen to be replicated. All variants provided "
                   "in the "
                << "leaf_variants must be leaf task variants.";
            error.raise();
          }
          // Check that the chosen variant is replicable
          if (!impl->is_replicable())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'replicate_task' "
                   "on mapper "
                << *mapper << ". Mapper failed to pick an valid task variant "
                << output.leaf_variants[idx] << " for " << *this
                << " that was chosen to be replicated. Task variants selected "
                   "for "
                << "replication must be marked as replicable variants.";
            error.raise();
          }
        }
      }
      if (runtime->safe_mapper)
      {
        // Check that all the processors exist
        bool has_local = false;
        for (unsigned idx = 0; idx < output.target_processors.size(); idx++)
        {
          if (!output.target_processors[idx].exists())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'replicate_task' "
                << "on mapper " << *mapper
                << ". Mapper specified a NO_PROC in the "
                << "vector of target processors when replicating " << *this
                << ". All processors in target_processors must exist.";
            error.raise();
          }
          else if (
              !has_local && (runtime->address_space ==
                             output.target_processors[idx].address_space()))
            has_local = true;
        }
        if (!has_local)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error
              << "Invalid mapper output from invocation of 'replicate_task' "
              << "on mapper " << *mapper << ". Mapper did not provide a local "
              << "processor when replicating " << *this
              << ". At last one shard "
              << "of a replicated task must be present on the node where the "
              << "task is being mapped to remain consistent with the semantics "
              << "of 'map_task' which would require this anyway even if the "
                 "task "
              << "were not replicated.";
          error.raise();
        }
        // Check that the chosen variant works with all the targets processors
        if (output.leaf_variants.empty())
        {
          const ProcessorConstraint& constraint =
              var_impl->execution_constraints.processor_constraint;
          if (constraint.is_valid())
          {
            for (unsigned idx = 0; idx < output.target_processors.size(); idx++)
              if (!constraint.can_use(output.target_processors[idx].kind()))
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error << "Invalid mapper output from invocation of "
                         "'replicate_task' "
                      << "on mapper " << *mapper << ". Mapper specified "
                      << output.target_processors[idx].kind() << " processor "
                      << output.target_processors[idx]
                      << " which cannot be used "
                      << "with variant " << output.chosen_variant << " when "
                      << "replicating " << *this << " as the variant does "
                      << "not support that kind of processor.";
                error.raise();
              }
          }
        }
        else
        {
          for (unsigned idx = 0; idx < output.target_processors.size(); idx++)
          {
            VariantImpl* impl = runtime->find_variant_impl(
                task_id, output.leaf_variants[idx], false /*can_fail*/);
            const ProcessorConstraint& constraint =
                impl->execution_constraints.processor_constraint;
            if (constraint.is_valid() &&
                !constraint.can_use(output.target_processors[idx].kind()))
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error
                  << "Invalid mapper output from invocation of "
                     "'replicate_task' "
                  << "on mapper " << *mapper << ". Mapper specified "
                  << output.target_processors[idx].kind() << " processor "
                  << output.target_processors[idx]
                  << " which cannot be used with variant "
                  << output.leaf_variants[idx] << " when replicating " << *this
                  << " as the variant does not support that kind of processor.";
              error.raise();
            }
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
            if (sorted_procs[idx - 1] == sorted_procs[idx])
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Invalid mapper output from invocation of "
                       "'replicate_task' "
                    << "on mapper " << *mapper
                    << ". Mapper provided duplicate target "
                    << "processors for non-leaf task variant "
                    << output.chosen_variant << " when replicating " << *this
                    << ". In order to control "
                    << "replicate a task all the target processors must be "
                       "unique.";
              error.raise();
            }
        }
        // Check that shard points match the size target processors if not empty
        if (!output.shard_points.empty())
        {
          if (!output.shard_domain.exists())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'replicate_task' "
                << "on mapper " << *mapper
                << ". Mapper provided shard_points without "
                << "providing an associated shard_domain when replicating "
                << *this
                << ". A shard domain must also be provided in conjunction with "
                   "a "
                << "set of shard points.";
            error.raise();
          }
          if (output.shard_points.size() != output.target_processors.size())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'replicate_task' "
                << "on mapper " << *mapper << ". Mapper provided "
                << output.shard_points.size() << " shard_points which does not "
                << "match the " << output.target_processors.size()
                << " target processors specified when replicating " << *this
                << ". If shard_points are provided they must exactly match the "
                << "number of target processors.";
            error.raise();
          }
          std::vector<DomainPoint> sorted_points = output.shard_points;
          std::sort(sorted_points.begin(), sorted_points.end());
          for (unsigned idx = 1; idx < sorted_points.size(); idx++)
            if (sorted_points[idx - 1] == sorted_points[idx])
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Invalid mapper output from invocation of "
                       "'replicate_task' "
                    << "on mapper " << *mapper
                    << ". Mapper provided duplicate shard "
                    << "points when replicating " << *this
                    << ". In order to control "
                    << "replicate a task all the target processors must be "
                       "unique.";
              error.raise();
            }
        }
        // Check that shard domain volume matches number of points if not empty
        if (output.shard_domain.exists())
        {
          if (output.shard_points.empty())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'replicate_task' "
                << "on mapper " << *mapper
                << ". Mapper provided shard_domain without "
                << "providing any associated shard_points when replicating"
                << *this
                << ". The shard_points data structure must also be populated "
                   "in "
                << "conjunction with a shard_domain.";
            error.raise();
          }
          if (output.shard_points.size() != output.shard_domain.get_volume())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'replicate_task' "
                << "on mapper " << *mapper << ". Mapper provided "
                << output.shard_points.size() << " shard_points for "
                << "shard_domain with " << output.shard_domain.get_volume()
                << " points when replicating " << *this << ". The number of "
                << "shard_points must exactly match the volume of the "
                   "shard_domain.";
            error.raise();
          }
          for (unsigned idx = 0; idx < output.shard_points.size(); idx++)
            if (!output.shard_domain.contains(output.shard_points[idx]))
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Invalid mapper output from invocation of "
                       "'replicate_task' "
                    << "on mapper " << *mapper
                    << ". Mapper provided a point in shard_points "
                    << "that is not contained in the shard_domain when "
                       "replicating "
                    << *this
                    << ". Each point in shard_points must exist in the "
                       "shard_domain.";
              error.raise();
            }
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
        std::map<DomainPoint, ShardID> shard_mapping;
        const int dim = output.shard_points.front().get_dim();
        if (dim != 1)
          isomorphic_points = false;
        for (unsigned idx = 0; idx < output.shard_points.size(); idx++)
        {
          if (isomorphic_points && (output.shard_points[idx][0] != idx))
            isomorphic_points = false;
          if (output.shard_points[idx].get_dim() != dim)
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Mapper " << *mapper
                  << " specified shard points with different "
                  << "dimensionalities of " << dim << " and "
                  << output.shard_points[idx].get_dim()
                  << " for 'replicate_task' call for " << *this
                  << ". All shard points must have the same dimenstionality.";
            error.raise();
          }
          std::pair<std::map<DomainPoint, ShardID>::iterator, bool> result =
              shard_mapping.insert(std::pair<DomainPoint, ShardID>(
                  output.shard_points[idx], idx));
          if (!result.second)
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Mapper " << *mapper << " specified duplicate shard point "
                  << "names for shards " << result.first->second << " and "
                  << idx << " in 'replicate_task' mapper call for " << *this
                  << ". Each shard point must be given a unique name.";
            error.raise();
          }
        }
        for (const std::pair<const DomainPoint, ShardID>& shard_pair :
             shard_mapping)
        {
          sorted_points.emplace_back(shard_pair.first);
          shard_lookup.emplace_back(shard_pair.second);
        }
        const int domain_dim = output.shard_domain.get_dim();
        if ((domain_dim > 0) && (domain_dim != dim))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error
              << "Mapper " << *mapper
              << " specified a 'shard_domain' output with dimensionality "
              << domain_dim << " different than the " << dim
              << " dimension points in 'shard_points' in 'replicate_task' call "
                 "for "
              << *this << ". The dimensionality of 'shard_domain' must "
              << "match the dimensionality of the 'shard_points'.";
          error.raise();
        }
      }
      else
      {
        // Mapper didn't specify it so we can fill it in
        output.shard_domain = Domain(
            DomainPoint(0), DomainPoint(output.target_processors.size() - 1));
        output.shard_points.reserve(output.target_processors.size());
        for (unsigned idx = 0; idx < output.target_processors.size(); idx++)
        {
          output.shard_points.emplace_back(DomainPoint(idx));
          sorted_points.emplace_back(DomainPoint(idx));
          shard_lookup.emplace_back(idx);
        }
      }
      // Construct the collective mapping
      std::vector<AddressSpaceID> spaces(output.target_processors.size() + 1);
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
      CollectiveMapping* mapping =
          new CollectiveMapping(spaces, runtime->legion_collective_radix);
      const DistributedID manager_did = runtime->get_available_distributed_id();
      LegionSpy::log_replication(
          get_unique_id(), manager_did, !var_impl->is_leaf());
      legion_assert(shard_manager == nullptr);
      std::vector<ShardID> local_shards;
      for (ShardID idx = 0; idx < output.target_processors.size(); idx++)
      {
        const Processor processor = output.target_processors[idx];
        if (processor.address_space() != runtime->address_space)
          continue;
        local_shards.emplace_back(idx);
      }
      Mapper::ContextConfigOutput configuration;
      if (!var_impl->is_leaf())
        // Compute a context configuration that all the shards will
        // use for the control replicated context. We need to do this
        // here so that all the shards get the same settings
        configure_execution_context(configuration);
      shard_manager = new ShardManager(
          manager_did, mapping, local_shards.size(), configuration,
          is_top_level_task(), isomorphic_points, !var_impl->is_leaf(),
          output.shard_domain, std::move(output.shard_points),
          std::move(sorted_points), std::move(shard_lookup), this);
      shard_manager->add_base_gc_ref(SINGLE_TASK_REF);
      // Now create our local shards and start them mapping
      for (unsigned idx = 0; idx < local_shards.size(); idx++)
        shard_manager->create_shard(
            local_shards[idx], output.target_processors[local_shards[idx]],
            output.leaf_variants.empty() ?
                output.chosen_variant :
                output.leaf_variants[local_shards[idx]],
            parent_ctx, this);
      // Distribute the shard manager and launch the shards
      shard_manager->distribute_explicit(
          this, output.chosen_variant, output.target_processors,
          output.leaf_variants);
      return true;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::map_all_regions(
        MustEpochOp* must_epoch_op, const DeferMappingArgs* defer_args)
    //--------------------------------------------------------------------------
    {
      // Only do this the first or second time through
      if ((defer_args == nullptr) || (defer_args->invocation_count < 2))
      {
        if (request_valid_instances)
        {
          // If the mapper wants valid instances we first need to do our
          // versioning analysis and then call the mapper
          if ((defer_args == nullptr /*first invocation*/) ||
              (defer_args->invocation_count < 1))
          {
            const RtEvent version_ready_event =
                perform_versioning_analysis(false /*post mapper*/);
            if (version_ready_event.exists() &&
                !version_ready_event.has_triggered())
              return defer_perform_mapping(
                  version_ready_event, must_epoch_op, 1 /*invocation count*/);
          }
          // Now do the mapping call
          if (!invoke_mapper(must_epoch_op))
            return false;
        }
        else
        {
          // If the mapper doesn't need valid instances, we do the mapper
          // call first and then see if we need to do any versioning analysis
          if ((defer_args == nullptr /*first invocation*/) ||
              (defer_args->invocation_count < 1))
          {
            if (!invoke_mapper(must_epoch_op))
              return false;
            const RtEvent version_ready_event =
                perform_versioning_analysis(true /*post mapper*/);
            if (version_ready_event.exists() &&
                !version_ready_event.has_triggered())
              return defer_perform_mapping(
                  version_ready_event, must_epoch_op, 1 /*invocation count*/);
          }
        }
      }
      // See if we have a remote trace info to use, if we don't then make
      // our trace info and do the initialization
      const TraceInfo trace_info = is_remote() ?
                                       TraceInfo(this, remote_trace_recorder) :
                                       TraceInfo(this);
      // If we'r recording then record the replay map task
      if (is_recording())
        trace_info.record_replay_mapping(
            single_task_termination, TASK_OP_KIND, map_applied_conditions);
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
                untracked_valid_regions.begin(), untracked_valid_regions.end(),
                0);
            const bool check_collective =
                IS_COLLECTIVE(regions.front()) ||
                std::binary_search(
                    check_collective_regions.begin(),
                    check_collective_regions.end(), 0);
            region_preconditions.back() =
                physical_perform_updates_and_registration(
                    regions[0], version_infos[0], 0, init_precondition,
                    single_task_termination, physical_instances[0],
                    source_instances[0], PhysicalTraceInfo(trace_info, 0),
                    map_applied_conditions, check_collective, record_valid);
          }
        }
        else
        {
          unsigned read_only_count = 0;
          std::vector<unsigned> performed_regions;
          performed_regions.reserve(region_count);
          std::vector<UpdateAnalysis*> analyses(region_count, nullptr);
          std::vector<RtEvent> reg_pre(region_count, RtEvent::NO_RT_EVENT);
          for (unsigned idx = 0; idx < logical_regions.size(); idx++)
          {
            if (no_access_regions[idx])
              continue;
            VersionInfo& local_info = get_version_info(idx);
            // If we virtual mapped it, there is nothing to do
            if (virtual_mapped[idx])
              continue;
            performed_regions.emplace_back(idx);
            const bool record_valid = !std::binary_search(
                untracked_valid_regions.begin(), untracked_valid_regions.end(),
                idx);
            const bool check_collective =
                IS_COLLECTIVE(logical_regions[idx]) ||
                std::binary_search(
                    check_collective_regions.begin(),
                    check_collective_regions.end(), idx);
            // apply the results of the mapping to the tree
            reg_pre[idx] = physical_perform_updates(
                logical_regions[idx], local_info, idx, init_precondition,
                single_task_termination, physical_instances[idx],
                source_instances[idx], PhysicalTraceInfo(trace_info, idx),
                map_applied_conditions, analyses[idx], check_collective,
                record_valid);
            if (IS_READ_ONLY(logical_regions[idx]))
              read_only_count++;
          }
          // In order to avoid cycles when mapping multiple tasks in parallel
          // with read-only requirements, we need to guarantee that all
          // read-only copies are issued before we can perform any registrations
          // for the task that will be using their results.
          if (read_only_count > 1)
          {
            std::vector<RtEvent> read_only_preconditions;
            read_only_preconditions.reserve(read_only_count);
            std::vector<unsigned> read_only_regions;
            read_only_regions.reserve(read_only_count);
            for (const unsigned& region_idx : performed_regions)
            {
              if (!IS_READ_ONLY(logical_regions[region_idx]))
                continue;
              read_only_regions.emplace_back(region_idx);
              const RtEvent precondition = reg_pre[region_idx];
              if (precondition.exists())
                read_only_preconditions.emplace_back(precondition);
            }
            if (!read_only_preconditions.empty())
            {
              const RtEvent read_only_precondition =
                  Runtime::merge_events(read_only_preconditions);
              if (read_only_precondition.exists())
              {
                for (const unsigned& region_idx : read_only_regions)
                  reg_pre[region_idx] = read_only_precondition;
              }
            }
          }
          for (const unsigned& region_idx : performed_regions)
          {
            region_preconditions[region_idx] = physical_perform_registration(
                reg_pre[region_idx], analyses[region_idx],
                map_applied_conditions,
                logical_regions.is_output_created(region_idx));
          }
        }
        if (perform_postmap)
          perform_post_mapping(trace_info);
      }  // if (!regions.empty())
      if (is_recording())
      {
        legion_assert(output_regions.empty());
        const TraceInfo trace_info =
            is_remote() ? TraceInfo(this, remote_trace_recorder) :
                          TraceInfo(this);
        if (execution_fence_event.exists())
          region_preconditions.emplace_back(execution_fence_event);
        ApEvent ready_event =
            Runtime::merge_events(&trace_info, region_preconditions);
        if (execution_fence_event.exists())
          region_preconditions.pop_back();
        const TraceLocalID tlid = get_trace_local_id();
        if (!atomic_locks.empty())
          trace_info.record_reservations(
              tlid, atomic_locks, map_applied_conditions);
        trace_info.record_complete_replay(map_applied_conditions, ready_event);
      }
      if (remote_trace_recorder != nullptr)
      {
        if (remote_trace_recorder->remove_recorder_reference())
          delete remote_trace_recorder;
        remote_trace_recorder = nullptr;
      }
      if (must_epoch_op != nullptr)
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
      return true;
    }

    //--------------------------------------------------------------------------
    void SingleTask::perform_post_mapping(const TraceInfo& trace_info)
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
          local::FieldMaskMap<ReplicatedView> collectives;
          physical_premap_region(
              idx, regions[idx], get_version_info(idx), postmap_valid,
              collectives, map_applied_conditions);
          // No need to filter these because they are on the way out
          prepare_for_mapping(
              postmap_valid, collectives, input.valid_instances[idx],
              input.valid_collectives[idx]);
        }
        local::FieldMaskMap<ReplicatedView> no_collectives;
        prepare_for_mapping(
            physical_instances[idx], no_collectives, input.mapped_regions[idx],
            input.valid_collectives[idx]);
      }
      // Now we can do the mapper call
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_post_map_task(this, input, output);
      // Check and register the results
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (no_access_regions[idx] || virtual_mapped[idx])
          continue;
        if (output.chosen_instances[idx].empty())
          continue;
        RegionRequirement& req = regions[idx];
        if (req.is_restricted())
        {
          Warning warning;
          warning << "Mapper " << *mapper << " requested post mapping "
                  << "instances be created for region requirement " << idx
                  << "of " << *this << ", but this region requirement "
                  << "is restricted. The request is being ignored.";
          warning.raise();
          continue;
        }
        if (IS_NO_ACCESS(req))
        {
          Warning warning;
          warning << "Mapper " << *mapper << " requested post mapping "
                  << "instances be created for region requirement " << idx
                  << "of task " << *this << ", but this region requirement "
                  << "has NO_ACCESS privileges. The request is being ignored.";
          warning.raise();
          continue;
        }
        if (IS_REDUCE(req))
        {
          Warning warning;
          warning << "Mapper " << *mapper << "requested post mapping "
                  << "instances be created for region requirement " << idx
                  << "of " << *this << ", but this region requirement "
                  << "has REDUCE privileges. The request is being ignored.";
          warning.raise();
          continue;
        }
        // Convert the post-mapping
        InstanceSet result;
        RegionTreeID bad_tree = 0;
        std::vector<PhysicalManager*> unacquired;
        bool had_composite = physical_convert_postmapping(
            req, output.chosen_instances[idx], result, bad_tree,
            !runtime->safe_mapper ? nullptr : &acquired_instances, unacquired,
            runtime->safe_mapper);
        if (bad_tree > 0)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from 'postmap_task' invocation on "
                   "mapper "
                << *mapper << ". Mapper provided an instance from region tree "
                << bad_tree << " for use in satisfying region requirement "
                << idx << " of " << *this
                << " whose region is from region tree "
                << regions[idx].region.get_tree_id() << ".";
          error.raise();
        }
        if (!unacquired.empty())
        {
          for (PhysicalManager* const manager : unacquired)
          {
            if (acquired_instances.find(manager) == acquired_instances.end())
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Invalid mapper output from 'postmap_task' "
                    << "invocation on mapper " << *mapper
                    << ". Mapper selected "
                    << "physical instance for region requirement " << idx
                    << " of " << *this << " which has already "
                    << "been collected. If the mapper had properly "
                    << "acquired this instance as part of the mapper "
                    << "call it would have detected this. Please update the "
                    << "mapper to abide by proper mapping conventions.";
              error.raise();
            }
          }
          // If we did successfully acquire them, still issue the warning
          Warning warning;
          warning << "Mapper " << *mapper << " failed to acquires instances "
                  << "for region requirement " << idx << " of " << *this
                  << "in 'postmap_task' call. You may experience "
                  << "undefined behavior as a consequence.";
          warning.raise();
        }
        if (had_composite)
        {
          Warning warning;
          warning << "Mapper " << *mapper << " requested a virtual "
                  << "instance be created for region requirement " << idx
                  << " of " << *this << " for a post mapping. The "
                  << "request is being ignored.";
          warning.raise();
          continue;
        }
        if (runtime->safe_mapper)
        {
          std::vector<LogicalRegion> regions_to_check(1, regions[idx].region);
          for (unsigned check_idx = 0; check_idx < result.size(); check_idx++)
          {
            PhysicalManager* manager = result[check_idx].get_physical_manager();
            if (!manager->meets_regions(regions_to_check))
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error
                  << "Invalid mapper output from invocation of 'postmap_task' "
                  << "on mapper " << *mapper << ". Mapper specified an "
                  << "instance region requirement " << idx << " of " << *this
                  << " that does not meet the logical region requirement.";
              error.raise();
            }
          }
        }
        log_mapping_decision(idx, regions[idx], result, true /*postmapping*/);
        // TODO: Implement physical tracing for postmapped regions
        if (is_recording())
          std::abort();
        // Register this with a no-event so that the instance can
        // be used as soon as it is valid from the copy to it
        // We also use read-only privileges to ensure that it doesn't
        // invalidate the other valid instances
        const PrivilegeMode mode = regions[idx].privilege;
        regions[idx].privilege = LEGION_READ_ONLY;
        VersionInfo& local_version_info = get_version_info(idx);
        std::vector<PhysicalManager*> sources;
        if (!output.source_instances[idx].empty())
          physical_convert_sources(
              regions[idx], output.source_instances[idx], sources,
              runtime->safe_mapper ? &acquired_instances : nullptr);
        physical_perform_updates_and_registration(
            regions[idx], local_version_info, idx,
            single_task_termination /*wait for task to be done*/,
            ApEvent::NO_AP_EVENT /*done immediately*/, result, sources,
            PhysicalTraceInfo(trace_info, idx), map_applied_conditions,
            false /*check for collectives*/, false /*track effects*/);
        regions[idx].privilege = mode;
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::check_future_return_bounds(FutureInstance* instance) const
    //--------------------------------------------------------------------------
    {
      if (future_return_size && (*future_return_size < instance->size))
      {
        Provenance* provenance = get_provenance();
        Error error(LEGION_DYNAMIC_TYPE_EXCEPTION);
        if (provenance != nullptr)
          error << "Task " << *this << " used a task variant "
                << "with a maximum return size of " << *future_return_size
                << " but returned a result of " << instance->size << " bytes.";
        else
          error << "Task " << *this << " used a task variant with a maximum "
                << "return size of " << *future_return_size
                << " but returned a result of " << instance->size << " bytes.";
        error.raise();
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::create_leaf_memory_pools(
        VariantImpl* variant,
        std::map<std::pair<Memory, bool>, PoolBounds>& dynamic_pool_bounds)
    //--------------------------------------------------------------------------
    {
      if (dynamic_pool_bounds.empty() && variant->leaf_pool_bounds.empty())
      {
        // If we're a concurrent task or a collectively mapped task then we
        // still need to participate in the max-allreduce for allocating
        // unbounded pools
        if (concurrent_task || must_epoch_task ||
            !check_collective_regions.empty())
          order_collectively_mapped_unbounded_pools(0, false /*need result*/);
        return;
      }
      // Fill in the dynamic pool bounds with the static versions
      for (const std::pair<const std::pair<Memory::Kind, bool>, PoolBounds>&
               pool_pair : variant->leaf_pool_bounds)
      {
        // This might occur if we're doing origin mapping on a remote node
        // from where the task is going to ultimately run
        Machine::MemoryQuery query(runtime->machine);
        query.only_kind(pool_pair.first.first);
        query.best_affinity_to(target_proc);
        if (query.count() == 0)
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Unable to find a visible " << pool_pair.first.first
                << " memory from processor " << target_proc << " for " << *this
                << " for creating dynamic memory pool from static constraint.";
          error.raise();
        }
        const Memory target = query.first();
        const std::pair<Memory, bool> key(target, pool_pair.first.second);
        // Check to see if we also got a dynamic memory pool bound, if we
        // did then it needs to tighten what already existed
        std::map<std::pair<Memory, bool>, PoolBounds>::const_iterator finder =
            dynamic_pool_bounds.find(key);
        if (finder != dynamic_pool_bounds.end())
        {
          if (pool_pair.second.is_bounded())
          {
            MemoryManager* manager = runtime->find_memory_manager(target);
            if (finder->second.is_bounded())
            {
              const PoolBounds& static_bounds = pool_pair.second;
              const PoolBounds& dynamic_bounds = finder->second;
              if (static_bounds.size < dynamic_bounds.size)
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error
                    << "Mapper " << *mapper << "dynamically requested "
                    << dynamic_bounds.size << " bytes for pool in "
                    << manager->get_name() << " memory for " << *this
                    << ", but the selected variant " << variant->vid
                    << " specified a static bound of "
                    << dynamic_bounds.alignment << " bytes. "
                    << "Dynamically requested memory allocations must be "
                       "further "
                    << "refinements of the upper bounds provided by the chosen "
                    << "task variant.";
                error.raise();
              }
              else if (static_bounds.alignment < dynamic_bounds.alignment)
              {
                Error error(LEGION_MAPPER_EXCEPTION);
                error
                    << "Mapper " << *mapper << " dynamically requested a "
                    << "minimum alignment of " << dynamic_bounds.alignment
                    << " bytes for pool in " << manager->get_name()
                    << " memory for " << *this << ", but the selected variant "
                    << variant->vid
                    << " specified a static minimum alignment of "
                    << static_bounds.alignment
                    << " bytes. Dynamically requested "
                    << "memory allocations must be further refinements of the "
                    << "alignments provided by the chosen task variant.";
                error.raise();
              }
            }
            else
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error
                  << "Mapper " << *mapper
                  << " dynamically requested an unbounded pool in "
                  << manager->get_name() << " memory for " << *this
                  << ", but the selected variant " << variant->vid
                  << " specified a static bound of " << pool_pair.second.size
                  << " bytes. Dynamically requested memory allocations must be "
                     "further refinements of the upper bounds provided by the "
                     "chosen task variant.";
              error.raise();
            }
          }
          else if (!finder->second.is_bounded())
          {
            // Else if the static variant had no bounds we know that the
            // dynamic one is at least as tight as the static one
            if (runtime->runtime_warnings)
            {
              Warning warning;
              warning
                  << "Selected variant " << variant->get_name() << " of "
                  << *this << " was registered with an unbound memory pool for "
                  << pool_pair.first.first << " memory and mapper " << *mapper
                  << "failed to tighten the bound. Unbound memory pools are "
                  << "very bad for performance and we strongly encourage all "
                  << "users to avoid using them except for extenuating "
                  << "circumstances when the amount of dynamic memory required "
                  << "by a task is truly unbounded.";
              warning.raise();
            }
          }
        }
        else
        {
          // We don't allow strict unbounded memory pools to be used
          // along with concurrent index space tasks or tasks that are
          // going to perform collective mapping of region requirements
          if (pool_pair.second.scope == LEGION_STRICT_UNBOUNDED_POOL)
          {
            if (concurrent_task)
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Mapper " << *mapper << " selected variant "
                    << variant->get_name() << " which statically mandates a "
                    << "strict unbounded memory pool in "
                    << pool_pair.first.first << " memory for concurrent task "
                    << *this << ". Strict unbounded "
                    << "memory pools are not permitted to be used when "
                    << "mapping concurrent index space tasks because they "
                    << "can lead to deadlocks. Instead you should use either "
                    << "LEGION_INDEX_TASK_UNBOUNDED_POOL or "
                    << "LEGION_PERMISSIVE_UNBOUNDED_POOL scope or pick a "
                    << "different task variant for mapping this task.";
              error.raise();
            }
            else if (!check_collective_regions.empty())
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error
                  << "Mapper " << *mapper << " selected variant "
                  << variant->get_name() << " which statically mandates a "
                  << "strict unbounded memory pool in " << pool_pair.first.first
                  << " memory for task " << *this
                  << " while the mapper also requested "
                  << "collective mapping of " << check_collective_regions.size()
                  << " region requirements. "
                  << "Strict unbounded memory pools are not permitted to be "
                  << "used in conjunction with collective mapping because they "
                  << "can lead to deadlocks. Instead you should use "
                  << "LEGION_INDEX_TASK_UNBOUNDED_POOL or "
                  << "LEGION_PERMISSIVE_UNBOUNDED_POOL scope or pick a "
                  << "different task variant for mapping this task or "
                  << "opt not to perform collective mapping of any regions.";
              error.raise();
            }
          }
          if (!pool_pair.second.is_bounded() && runtime->runtime_warnings)
          {
            Warning warning;
            warning
                << "Selected variant " << variant->get_name() << " of task "
                << *this << " was registered with an unbound memory pool for "
                << pool_pair.first.first << " memory and mapper " << *mapper
                << "failed to tighten the bound. Unbound memory pools are "
                << "very bad for performance and we strongly encourage all "
                << "users to avoid using them except for extenuating "
                << "circumstances when the amount of dynamic memory required "
                << "by a task is truly unbounded.";
            warning.raise();
          }
          dynamic_pool_bounds.emplace(std::make_pair(key, pool_pair.second));
        }
      }
      // If we're a concurrent task or a collectively mapped task then we
      // need to go through and get lamport clocks for all the memories in
      // which we're planning to allocate unbounded pools to make sure that
      // our concurrent/collective mapped task is ordered with respect to
      // any other that are also trying to map in parallel with us
      RtEvent wait_for_unbounded_allocations;
      std::vector<MemoryManager*> unbounded_pools;
      if (concurrent_task || must_epoch_task ||
          !check_collective_regions.empty())
      {
        uint64_t max_lamport_clock = 0;
        for (const std::pair<const std::pair<Memory, bool>, PoolBounds>&
                 pool_pair : dynamic_pool_bounds)
        {
          if (pool_pair.second.is_bounded())
            continue;
          MemoryManager* manager =
              runtime->find_memory_manager(pool_pair.first.first);
          // We might want to relax this restriction in the future to allow
          // tasks to make deferred buffers on memories that are "remote"
          // from where they are executing
          if (pool_pair.first.first.address_space() !=
              target_proc.address_space())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << manager->get_name() << " memory " << pool_pair.first.first
                  << " is not visible from the target processor of " << *this
                  << " for creating dynamic memory pool.";
            error.raise();
          }
          unbounded_pools.emplace_back(manager);
          uint64_t lamport_clock =
              manager->order_collective_unbounded_pools(this);
          if (max_lamport_clock < lamport_clock)
            max_lamport_clock = lamport_clock;
        }
        // Perform the max-allreduce of this lamport clock across all the
        // point tasks in the index space launch
        if (!unbounded_pools.empty())
        {
          max_lamport_clock = order_collectively_mapped_unbounded_pools(
              max_lamport_clock, true /*need result*/);
          // Tell the memory manager about the max lamport clock and get
          // back any events that we need to wait on to know that it is
          // safe to start allocating unbounded pools
          std::vector<RtEvent> wait_for;
          for (MemoryManager* const manager : unbounded_pools)
          {
            RtEvent ready = manager->finalize_collective_unbounded_pools_order(
                this, max_lamport_clock);
            if (ready.exists())
              wait_for.emplace_back(ready);
          }
          unbounded_pools.clear();
          // Just record the event to wait on for now, we might be able
          // to do some bounded allocations in the meantime before we
          // actually need to block waiting
          if (!wait_for.empty())
            wait_for_unbounded_allocations = Runtime::merge_events(wait_for);
        }
        else
          // we don't have any unbounded pools so we still participate
          // but we don't need to block waiting for the result
          order_collectively_mapped_unbounded_pools(
              max_lamport_clock, false /*need result*/);
      }
      RtEvent safe_single_task_termination;
      std::map<std::pair<Memory, bool>, MemoryPool*> acquired_pools;
      acquired_pools.swap(leaf_memory_pools);
      // Now we can go through and create the pools for use by this task
      for (const std::pair<const std::pair<Memory, bool>, PoolBounds>&
               pool_pair : dynamic_pool_bounds)
      {
        MemoryManager* manager =
            runtime->find_memory_manager(pool_pair.first.first);
        // We might want to relax this restriction in the future to allow tasks
        // to make deferred buffers on memories that are "remote" from where
        // they are executing
        if (pool_pair.first.first.address_space() !=
            target_proc.address_space())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << manager->get_name() << " memory " << pool_pair.first.first
                << " is not visible from the target processor of " << *this
                << " for creating dynamic memory pool.";
          error.raise();
        }
        if (pool_pair.second.is_bounded())
        {
          // Check to see if acquired a memory pool for this already
          std::map<std::pair<Memory, bool>, MemoryPool*>::iterator finder =
              acquired_pools.find(pool_pair.first);
          if (finder != acquired_pools.end())
          {
            // These should all be bounded pools
            legion_assert(
                finder->second->get_bounds().scope == LEGION_BOUNDED_POOL);
            if (!pool_pair.first.second)
            {
              // Non-escaping so do the finalize now
              if (!safe_single_task_termination.exists())
              {
                legion_assert(single_task_termination.exists());
                safe_single_task_termination =
                    Runtime::protect_event(single_task_termination);
              }
              finder->second->finalize_pool(safe_single_task_termination);
            }
            leaf_memory_pools.insert(acquired_pools.extract(finder));
            continue;
          }
        }
        else
        {
          // We don't allow strict unbounded memory pools to be used
          // along with concurrent index space tasks or tasks that are
          // going to perform collective mapping of region requirements
          if (pool_pair.second.scope == LEGION_STRICT_UNBOUNDED_POOL)
          {
            if (concurrent_task)
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Mapper " << *mapper << " dynamically requested a "
                    << "strict unbounded memory pool in " << manager->get_name()
                    << " memory for concurrent task " << *this
                    << ". Strict unbounded "
                    << "memory pools are not permitted to be used when "
                    << "mapping concurrent index space tasks because they "
                    << "can lead to deadlocks. Instead you should use either "
                    << "LEGION_INDEX_TASK_UNBOUNDED_POOL or "
                    << "LEGION_PERMISSIVE_UNBOUNDED_POOL scope when specifying "
                    << "the kind of unbound memory pool for this task.";
              error.raise();
            }
            else if (!check_collective_regions.empty())
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error
                  << "Mapper " << *mapper << " dynamically requested a "
                  << "strict unbounded memory pool in " << manager->get_name()
                  << " memory for task " << *this << " while also requesting a "
                  << "collective mapping of " << check_collective_regions.size()
                  << " region requirements. "
                  << "Strict unbounded memory pools are not permitted to be "
                  << "used in conjunction with collective mapping because they "
                  << "can lead to deadlocks. Instead you should use "
                  << "LEGION_INDEX_TASK_UNBOUNDED_POOL or "
                  << "LEGION_PERMISSIVE_UNBOUNDED_POOL scope for any unbounded "
                  << "memory pools for this task or alternatively choosen not "
                  << "to perform collective mapping of any region "
                     "requirements.";
              error.raise();
            }
            if (runtime->runtime_warnings)
            {
              const std::pair<Memory::Kind, bool> key(
                  pool_pair.first.first.kind(), true /*escaping*/);
              if (variant->leaf_pool_bounds.find(key) ==
                  variant->leaf_pool_bounds.end())
              {
                Warning warning;
                warning
                    << "Mapper " << *mapper
                    << " requested an unbound memory pool in "
                    << manager->get_name() << " memory for leaf task " << *this
                    << ". Unbound memory pools are very bad "
                    << "for performance and we strongly encourage all users to "
                       "avoid"
                    << " using them except for extenuating circumstances when "
                       "the "
                    << "amount of dynamic memory required by a task is truly "
                       "unbounded.";
                warning.raise();
              }
            }
          }
          // If this is our first unbounded allocation then we need to
          // wait for it to safe for us to do it
          if (wait_for_unbounded_allocations.exists())
          {
            wait_for_unbounded_allocations.wait();
            wait_for_unbounded_allocations = RtEvent::NO_RT_EVENT;
          }
        }
        // It's not safe to hold onto unbounded pools if we're going to
        // block on trying to allocate something else as this can lead to
        // hangs associated with things like concurrent index space task
        // launches that need to be able to acquire all their pools without
        // us interleaving in between them.
        RtEvent try_again;
        MemoryPool* pool = nullptr;
        unsigned unbound_acquired_index = unbounded_pools.size();
        do {
          if (try_again.exists())
          {
            // Release all unbounded pools
            for (unsigned idx = 0; idx < unbound_acquired_index; idx++)
            {
              std::map<std::pair<Memory, bool>, MemoryPool*>::iterator finder =
                  leaf_memory_pools.find(
                      std::make_pair(unbounded_pools[idx]->memory, true));
              legion_assert(finder != leaf_memory_pools.end());
              delete finder->second;
              leaf_memory_pools.erase(finder);
            }
            // Wait to try again
            try_again.wait();
            try_again = RtEvent::NO_RT_EVENT;
            // Try to reacquire all unbounded pools
            for (unbound_acquired_index = 0;
                 unbound_acquired_index < unbounded_pools.size();
                 unbound_acquired_index++)
            {
              MemoryManager* target = unbounded_pools[unbound_acquired_index];
              std::map<std::pair<Memory, bool>, PoolBounds>::const_iterator
                  finder = dynamic_pool_bounds.find(
                      std::make_pair(target->memory, true));
              legion_assert(finder != dynamic_pool_bounds.end());
              legion_assert(!finder->second.is_bounded());
              // Recompute these each time as they might be consumed each time
              TaskTreeCoordinates coordinates;
              compute_task_tree_coordinates(coordinates);
              MemoryPool* unbound_pool = target->create_memory_pool(
                  get_unique_id(), coordinates, finder->second, &try_again);
              if (try_again.exists())
                break;
              leaf_memory_pools.emplace(std::make_pair(
                  std::make_pair(target->memory, true), unbound_pool));
            }
            legion_assert(
                try_again.exists() ==
                (unbound_acquired_index < unbounded_pools.size()));
            if (try_again.exists())
              continue;
          }
          // Try to do the most recent allocation
          // Recompute these each time as they might be consumed each time
          TaskTreeCoordinates coordinates;
          compute_task_tree_coordinates(coordinates);
          // Not safe to block here indefinitely holding unbounded pools
          pool = manager->create_memory_pool(
              get_unique_id(), coordinates, pool_pair.second, &try_again);
        } while (try_again.exists());
        if (pool == nullptr)
        {
          Error error(LEGION_RESOURCE_EXCEPTION);
          error
              << "Failed to reserve a dynamic memory pool of "
              << pool_pair.second.size << " bytes for leaf task " << *this
              << " in " << manager->get_name()
              << " memory. You are actually out "
              << "of memory here so you'll need to either allocate more memory "
              << "for this kind of memory when you configure Realm which may "
              << "necessitate finding a bigger machine. If you want to avoid "
              << "finding out that you're out of memory this way you should "
              << "instead use the 'MapperRuntime::acquire_pool' call to make "
              << "sure that memory can be reserved for all pools in advance.";
          error.raise();
        }
        if (!pool_pair.first.second)
        {
          // Non escaping pools need to be finalized early
          if (!safe_single_task_termination.exists())
          {
            legion_assert(single_task_termination.exists());
            safe_single_task_termination =
                Runtime::protect_event(single_task_termination);
          }
          pool->finalize_pool(safe_single_task_termination);
        }
        leaf_memory_pools.emplace(std::make_pair(pool_pair.first, pool));
        // Keep track of all our unbounded pools
        if (!pool_pair.second.is_bounded())
          unbounded_pools.emplace_back(manager);
      }
      if (concurrent_task || must_epoch_task ||
          !check_collective_regions.empty())
      {
        // Tell our unbounded pools that we're done allocating
        for (MemoryManager* manager : unbounded_pools)
          manager->end_collective_unbounded_pools_task();
      }
      // If we have any pools left in the acquired set we can delete them
      // since we're not going to need them
      for (const std::pair<const std::pair<Memory, bool>, MemoryPool*>&
               pool_pair : acquired_pools)
        delete pool_pair.second;
      acquired_pools.clear();
    }

    //--------------------------------------------------------------------------
    bool SingleTask::acquire_leaf_memory_pool(
        Memory memory, const PoolBounds& bounds, bool escaping, bool optimistic,
        RtEvent* safe_for_unbounded_pools)
    //--------------------------------------------------------------------------
    {
      legion_assert(bounds.is_bounded());
      const std::pair<Memory, bool> key(memory, escaping);
      // Check to see if we already have a memory pool for this memory of
      // the given size, if we do then we're already good
      std::map<std::pair<Memory, bool>, MemoryPool*>::iterator finder =
          leaf_memory_pools.find(key);
      if (finder != leaf_memory_pools.end())
      {
        if ((bounds.size <= finder->second->query_available_memory()) &&
            (bounds.alignment <= finder->second->max_alignment))
          return true;
        if (optimistic)
        {
          // Otherwise release this pool since we're going to make a new one
          delete finder->second;
          leaf_memory_pools.erase(finder);
        }
      }
      TaskTreeCoordinates coordinates;
      compute_task_tree_coordinates(coordinates);
      MemoryManager* manager = runtime->find_memory_manager(memory);
      MemoryPool* pool = manager->create_memory_pool(
          get_unique_id(), coordinates, bounds, safe_for_unbounded_pools);
      if (pool == nullptr)
        return false;
      if (!optimistic && (finder != leaf_memory_pools.end()))
      {
        delete finder->second;
        finder->second = pool;
      }
      else
        leaf_memory_pools[key] = pool;
      return true;
    }

    //--------------------------------------------------------------------------
    void SingleTask::release_leaf_memory_pool(Memory memory, bool escaping)
    //--------------------------------------------------------------------------
    {
      const std::pair<Memory, bool> key(memory, escaping);
      std::map<std::pair<Memory, bool>, MemoryPool*>::iterator finder =
          leaf_memory_pools.find(key);
      if (finder != leaf_memory_pools.end())
      {
        delete finder->second;
        leaf_memory_pools.erase(key);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      legion_assert(logical_regions.size() == physical_instances.size());
      legion_assert(logical_regions.size() == no_access_regions.size());
      // If we haven't computed our virtual mapping information
      // yet (e.g. because we origin mapped) then we have to
      // do that now
      if (virtual_mapped.empty())
      {
        virtual_mapped.resize(regions.size(), false);
        for (unsigned idx = 0; idx < regions.size(); idx++)
          virtual_mapped[idx] = physical_instances[idx].is_virtual_mapping();
      }
      VariantImpl* variant =
          runtime->find_variant_impl(task_id, selected_variant);
      // STEP 1: Compute the precondition for the task launch
      std::set<ApEvent> wait_on_events;
      if (execution_fence_event.exists())
        wait_on_events.insert(execution_fence_event);
      // TODO: teach legion spy how to check the inner task optimization
      // for now we'll just turn it off whenever we are going to be
      // validating the runtime analysis
      const bool do_inner_task_optimization =
          (spy_logging_level > LIGHT_SPY_LOGGING) ? false : variant->is_inner();
      // Get the event to wait on unless we are
      // doing the inner task optimization
      if (!do_inner_task_optimization)
      {
        for (unsigned idx = 0; idx < regions.size(); idx++)
          if (region_preconditions[idx].exists())
            wait_on_events.insert(region_preconditions[idx]);
        for (unsigned idx = 0; idx < futures.size(); idx++)
        {
          FutureImpl* impl = futures[idx].impl;
          if (impl == nullptr)
            continue;
          ApEvent ready;
          if ((idx < future_memories.size()) && future_memories[idx].exists())
            ready = impl->find_application_instance_ready(
                future_memories[idx], this);
          else
            ready = impl->get_complete_event();
          if (ready.exists())
            wait_on_events.insert(ready);
        }
      }
      // Now add get all the other preconditions for the launch
      for (const Grant& grant : grants)
      {
        GrantImpl* impl = grant.impl;
        wait_on_events.insert(impl->acquire_grant());
      }
      for (const PhaseBarrier& barrier : wait_barriers)
      {
        ApEvent e = Runtime::get_previous_phase(barrier.phase_barrier);
        wait_on_events.insert(e);
      }

      // STEP 2: Set up the task's context
      std::vector<ApUserEvent> unmap_events(regions.size());
      {
        const bool is_leaf_variant = variant->is_leaf();
        execution_context = create_execution_context(
            variant, wait_on_events, inline_task, is_leaf_variant);
        std::vector<RegionRequirement> clone_requirements(regions.size());
        // Make physical regions for each our region requirements
        for (unsigned idx = 0; idx < regions.size(); idx++)
        {
          legion_assert(regions[idx].handle_type == LEGION_SINGULAR_PROJECTION);
          log_mapping_decision(idx, regions[idx], physical_instances[idx]);
          // If it was virtual mapper so it doesn't matter anyway.
          if (virtual_mapped[idx] || no_access_regions[idx])
          {
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            execution_context->add_physical_region(
                clone_requirements[idx], false /*mapped*/, map_id, tag,
                unmap_events[idx], true /*virtual mapped*/,
                physical_instances[idx]);
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
            execution_context->add_physical_region(
                clone_requirements[idx], false /*mapped*/, map_id, tag,
                unmap_events[idx], false /*virtual mapped*/,
                physical_instances[idx]);
            legion_assert(unmap_events[idx].exists());
            // Trigger the user event when the region is
            // actually ready to be used
            Runtime::trigger_event_untraced(
                unmap_events[idx], region_preconditions[idx]);
          }
          else
          {
            // If this is not virtual mapped, here is where we
            // switch coherence modes from whatever they are in
            // the enclosing context to exclusive within the
            // context of this task
            clone_requirements[idx] = regions[idx];
            localize_region_requirement(clone_requirements[idx]);
            execution_context->add_physical_region(
                clone_requirements[idx], true /*mapped*/, map_id, tag,
                unmap_events[idx], false /*virtual mapped*/,
                physical_instances[idx]);
            // We reset the reference below after we've
            // initialized the local contexts and received
            // back the local instance references
          }
        }
        // Initialize output regions
        for (unsigned idx = 0; idx < output_regions.size(); ++idx)
        {
          log_mapping_decision(
              regions.size() + idx, output_regions[idx],
              physical_instances[regions.size() + idx]);
          execution_context->add_output_region(
              output_regions[idx], physical_instances[regions.size() + idx],
              is_output_global(idx), is_output_valid(idx),
              is_output_grouped(idx));
        }
        // Initialize any region tree contexts
        execution_context->initialize_region_tree_contexts(
            clone_requirements, unmap_events);
        // Update the physical regions with any padding they might have
        if (variant->needs_padding)
          execution_context->record_padded_fields(variant);
      }
      // If we have a predicate event then merge that in here as well
      if (true_guard.exists())
        wait_on_events.insert(ApEvent(true_guard));
      // Merge together all the events for the start condition
      ApEvent start_condition = Runtime::merge_events(nullptr, wait_on_events);
      // If we're performing a concurrent index space task launch then we
      // need to perform an extra step here to ensure a global ordering
      // between concurrent index space task launches on the same processor
      if (is_concurrent())
        start_condition = order_concurrent_launch(start_condition, variant);
      // Need a copy of any locks to release on the stack since the
      // atomic_locks cannot be touched after we launch the task
      std::vector<Reservation> to_release;
      if (!atomic_locks.empty())
      {
        // Take all the locks in order in the proper way
        to_release.reserve(atomic_locks.size());
        for (const std::pair<const Reservation, bool>& lock_pair : atomic_locks)
        {
          start_condition = Runtime::acquire_ap_reservation(
              lock_pair.first, lock_pair.second, start_condition);
          to_release.emplace_back(lock_pair.first);
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
      legion_assert(!target_processors.empty());
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
        for (const ProfilingMeasurementID& measurement_id :
             task_profiling_requests)
        {
          if (measurement_id < Mapping::PMID_LEGION_FIRST)
            realm_measurements.insert(
                (Realm::ProfilingMeasurementID)measurement_id);
          else if (measurement_id == Mapping::PMID_RUNTIME_OVERHEAD)
            execution_context->initialize_overhead_profiler();
          else
            std::abort();  // should never get here
        }
        if (!realm_measurements.empty())
        {
          // If we're doing profiling we need the fevent to know how to
          // profile the profiling response
          if (runtime->profiler != nullptr)
            realm_measurements.insert(Realm::PMID_OP_FINISH_EVENT);
          OpProfilingResponse response(
              this, 0, 0, false /*fill*/, true /*task*/);
          Realm::ProfilingRequest& request = profiling_requests.add_request(
              runtime->find_utility_group(), LG_LEGION_PROFILING_ID, &response,
              sizeof(response));
          request.add_measurements(realm_measurements);
          // No need to increment the number of outstanding profiling
          // requests here since it was already done when we invoked
          // the mapper (see SingleTask::invoke_mapper)
          // The exeception is for origin-mapped remote tasks on which
          // we're going to need to send a message back to the owner
          if (is_remote() && is_origin_mapped())
          {
            legion_assert(outstanding_profiling_requests == 0);
            legion_assert(!profiling_reported.exists());
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
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        LegionSpy::log_variant_decision(unique_op_id, selected_variant);
        LegionSpy::log_operation_events(
            unique_op_id, start_condition, single_task_termination);
        // Chain the start event into the unmap events so Legion Spy can see
        // the dependences between child operations the start of the parent task
        for (const ApUserEvent& event : unmap_events)
          if (event.exists())
            LegionSpy::log_event_dependence(start_condition, event);
        LegionSpy::log_task_priority(unique_op_id, task_priority);
        for (const Future& future : futures)
        {
          FutureImpl* impl = future.impl;
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
        legion_assert(single_task_termination.exists());
        chain_task_termination = single_task_termination;
      }
      ApEvent task_launch_event = variant->dispatch_task(
          launch_processor, this, execution_context, start_condition,
          task_priority, profiling_requests);
      // Release any reservations that we took on behalf of this task
      // Note this happens before protection of the event for predication
      // because the acquires were also subject to poisoning so we either
      // want all the releases to be done or poisoned the same as the acquires
      if (!to_release.empty())
      {
        for (const Reservation& reservation : to_release)
          Runtime::release_reservation(reservation, task_launch_event);
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
          task_launch_event = Runtime::merge_events(nullptr, wait_on_events);
        }
        // Protect the single task termination from the poison
        if (chain_task_termination.exists())
          Runtime::trigger_event_untraced(
              chain_task_termination, Runtime::ignorefaults(task_launch_event));
      }
      else if (chain_task_termination.exists())
        Runtime::trigger_event_untraced(
            chain_task_termination, task_launch_event);
      // Finally if this is a predicated task and we have a speculative
      // guard then we need to launch a meta task to handle the case
      // where the task misspeculates
      if (misspeculation_precondition.exists())
      {
        MispredicationTaskArgs args(this);
        // Make sure this runs on an application processor where the
        // original task was going to go
        runtime->issue_runtime_meta_task(
            args, LG_LATENCY_WORK_PRIORITY, misspeculation_precondition);
        // Fun little trick here: decrement the outstanding meta-task
        // counts for the mis-speculation task in case it doesn't run
        // If it does run, we'll increment the counts again
        runtime->decrement_total_outstanding_tasks(
            MispredicationTaskArgs::TASK_ID, true /*meta*/);
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
        runtime->outstanding_counts[MispredicationTaskArgs::TASK_ID].fetch_sub(
            1);
#endif
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::pack_profiling_requests(
        Serializer& rez, std::set<RtEvent>& applied) const
    //--------------------------------------------------------------------------
    {
      rez.serialize(copy_fill_priority);
      rez.serialize<size_t>(copy_profiling_requests.size());
      if (!copy_profiling_requests.empty())
      {
        for (const ProfilingMeasurementID& request : copy_profiling_requests)
          rez.serialize(request);
        rez.serialize(profiling_priority);
        rez.serialize(runtime->find_utility_group());
        // Send a message to the owner with an update for the extra counts
        const RtUserEvent done_event = Runtime::create_rt_user_event();
        rez.serialize<RtEvent>(done_event);
        applied.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    int SingleTask::add_copy_profiling_request(
        const PhysicalTraceInfo& info, Realm::ProfilingRequestSet& requests,
        bool fill, unsigned count)
    //--------------------------------------------------------------------------
    {
      // Nothing to do if we don't have any copy profiling requests
      if (copy_profiling_requests.empty())
        return copy_fill_priority;
      OpProfilingResponse response(this, info.index, info.dst_index, fill);
      Realm::ProfilingRequest& request = requests.add_request(
          runtime->find_utility_group(), LG_LEGION_PROFILING_ID, &response,
          sizeof(response));
      bool has_finish = false;
      for (const ProfilingMeasurementID& measurement_id :
           copy_profiling_requests)
      {
        const Realm::ProfilingMeasurementID measurement =
            (Realm::ProfilingMeasurementID)measurement_id;
        request.add_measurement(measurement);
        if (measurement == Realm::PMID_OP_FINISH_EVENT)
          has_finish = true;
      }
      // Need thetimeline for the operation to know how to profile this
      // profiling response
      if (!has_finish && (runtime->profiler != nullptr))
        request.add_measurement(Realm::PMID_OP_FINISH_EVENT);
      handle_profiling_update(count);
      return copy_fill_priority;
    }

    //--------------------------------------------------------------------------
    bool SingleTask::handle_profiling_response(
        const Realm::ProfilingResponse& response, const void* orig,
        size_t orig_length, LgEvent& fevent, bool& failed_alloc)
    //--------------------------------------------------------------------------
    {
      legion_assert(profiling_reported.exists());
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      const OpProfilingResponse* task_prof =
          static_cast<const OpProfilingResponse*>(response.user_data());
      Realm::ProfilingMeasurements::OperationFinishEvent finish_event;
      if (response.get_measurement(finish_event))
        fevent = LgEvent(finish_event.finish_event);
      // First see if this is a task response for an origin-mapped task
      // on a remote node that needs to be sent back to the origin node
      if (task_prof->task && is_origin_mapped() && is_remote())
      {
        // We need to send this response back to the owner node along
        // with the overhead tracker
        SingleTask* orig_task = get_origin_task();
        RemoteTaskProfilingResponse rez;
        {
          RezCheck z(rez);
          rez.serialize(orig_task);
          rez.serialize(orig_length);
          rez.serialize(orig, orig_length);
          if (execution_context->overhead_profiler)
          {
            rez.serialize<bool>(true);
            // Only pack the bits that we need for the profiling response
            rez.serialize(
                (const void*)execution_context->overhead_profiler,
                sizeof(Mapping::ProfilingMeasurements::RuntimeOverhead));
          }
          else
            rez.serialize<bool>(false);
        }
        rez.dispatch(orig_proc.address_space());
      }
      else
      {
        // Check to see if we are done mapping, if not then we need to defer
        // this until we are done mapping so we know how many reports to expect
        const RtEvent mapped = get_mapped_event();
        if (!mapped.has_triggered())
          mapped.wait();
        // If we get here then we can handle the response now
        Mapping::Mapper::TaskProfilingInfo info;
        info.profiling_responses.attach_realm_profiling_response(response);
        info.task_response = task_prof->task;
        info.region_requirement_index = task_prof->src;
        info.total_reports = outstanding_profiling_requests;
        info.fill_response = task_prof->fill;
        if (info.task_response)
        {
          // If we had an overhead profiler
          // see if this is the callback for the task
          if (execution_context->overhead_profiler != nullptr)
            // This is the callback for the task itself
            info.profiling_responses.attach_overhead(
                execution_context->overhead_profiler);
        }
        mapper->invoke_task_report_profiling(this, info);
      }
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
      legion_assert(count <= outstanding_profiling_requests);
      if (count == outstanding_profiling_requests)
        Runtime::trigger_event(profiling_reported);
      // Always record these as part of profiling
      return true;
    }

    //--------------------------------------------------------------------------
    void SingleTask::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      legion_assert(count > 0);
      legion_assert(!mapped_event.has_triggered());
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void SingleTask::finalize_single_task_profiling(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(profiling_reported.exists());
      if (outstanding_profiling_requests == 0)
      {
        if (mapper == nullptr)
          mapper = runtime->find_mapper(current_proc, map_id);
        // We're not expecting any profiling callbacks so we need to
        // do one ourself to inform the mapper that there won't be any
        Mapping::Mapper::TaskProfilingInfo info;
        info.total_reports = 0;
        info.task_response = true;
        info.region_requirement_index = 0;
        info.fill_response = false;  // make valgrind happy
        mapper->invoke_task_report_profiling(this, info);
        Runtime::trigger_event(profiling_reported);
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::handle_remote_profiling_response(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      size_t buffer_size;
      derez.deserialize(buffer_size);
      const void* buffer = derez.get_current_pointer();
      derez.advance_pointer(buffer_size);
      // Realm needs this buffer to have 8-byte alignment so check that it does
      legion_assert((uintptr_t(buffer) % 8) == 0);
      bool has_tracker;
      derez.deserialize(has_tracker);
      Mapping::ProfilingMeasurements::RuntimeOverhead tracker;
      if (has_tracker)
        derez.deserialize(tracker);
      const Realm::ProfilingResponse response(buffer, buffer_size);
      const OpProfilingResponse* task_prof =
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
      legion_assert(count <= outstanding_profiling_requests.load());
      if (count == outstanding_profiling_requests.load())
        Runtime::trigger_event(profiling_reported);
    }

    //--------------------------------------------------------------------------
    /*static*/ void RemoteTaskProfilingResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      SingleTask* target;
      derez.deserialize(target);
      target->handle_remote_profiling_response(derez);
    }

    //--------------------------------------------------------------------------
    TaskContext* SingleTask::create_execution_context(
        VariantImpl* v, std::set<ApEvent>& launch_events, bool inline_task,
        bool leaf_task)
    //--------------------------------------------------------------------------
    {
      if (!leaf_task)
      {
        Mapper::ContextConfigOutput configuration;
        configure_execution_context(configuration);
        InnerContext* inner_ctx = nullptr;
        if (configuration.auto_tracing_enabled)
        {
          log_auto_trace.info(
              "Initializing auto tracing for %s (UID %lld)", get_task_name(),
              get_unique_id());
          inner_ctx = new AutoTracing<InnerContext>(
              configuration, this, get_depth(), v->is_inner(), regions,
              output_regions, parent_req_indexes, virtual_mapped, task_priority,
              execution_fence_event, 0 /*did*/, inline_task, false /*implicit*/,
              concurrent_task || parent_ctx->is_concurrent_context());
        }
        else
          inner_ctx = new InnerContext(
              configuration, this, get_depth(), v->is_inner(), regions,
              output_regions, parent_req_indexes, virtual_mapped, task_priority,
              execution_fence_event, 0 /*did*/, inline_task, false /*implicit*/,
              concurrent_task || parent_ctx->is_concurrent_context());
        inner_ctx->add_base_gc_ref(SINGLE_TASK_REF);
        return inner_ctx;
      }
      else
      {
        for (const std::pair<const std::pair<Memory, bool>, MemoryPool*>&
                 pool_pair : leaf_memory_pools)
        {
          const ApEvent ready = pool_pair.second->get_ready_event();
          if (ready.exists())
            launch_events.insert(ready);
        }
        LeafContext* leaf_ctx =
            new LeafContext(this, std::move(leaf_memory_pools), inline_task);
        leaf_memory_pools.clear();
        leaf_ctx->add_base_gc_ref(SINGLE_TASK_REF);
        return leaf_ctx;
      }
    }

    //--------------------------------------------------------------------------
    void SingleTask::record_inner_termination(ApEvent termination_event)
    //--------------------------------------------------------------------------
    {
      if (single_task_termination.exists())
        Runtime::trigger_event_untraced(
            single_task_termination, termination_event);
      else  // happens with implicit top-level tasks
        record_completion_effect(termination_event);
    }

    //--------------------------------------------------------------------------
    void SingleTask::OrderConcurrentLaunchArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      runtime->order_concurrent_task_launch(processor, task, start, ready, vid);
    }

  }  // namespace Internal
}  // namespace Legion
