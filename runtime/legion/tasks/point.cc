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

#include "legion/tasks/point.h"
#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/nodes/index.h"
#include "legion/operations/mustepoch.h"
#include "legion/tasks/index.h"
#include "legion/tasks/slice.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Point Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    PointTask::PointTask(void) : SingleTask()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    PointTask::~PointTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void PointTask::activate(void)
    //--------------------------------------------------------------------------
    {
      SingleTask::activate();
      orig_task = this;
      slice_owner = nullptr;
      concurrent_color = 0;
      concurrent_task_barrier = RtBarrier::NO_RT_BARRIER;
    }

    //--------------------------------------------------------------------------
    void PointTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (implicit_profiler != nullptr)
        implicit_profiler->register_slice_owner(
            this->slice_owner->get_unique_op_id(), this->get_unique_op_id());
      pointwise_mapping_dependences.clear();
      SingleTask::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
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
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      tpl->register_operation(this);
      SingleTask::trigger_replay();
      slice_owner->record_point_mapped(this, get_mapped_event());
    }

    //--------------------------------------------------------------------------
    void PointTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool PointTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_origin_mapped());
      // Point tasks are never sent anywhere
      return true;
    }

    //--------------------------------------------------------------------------
    bool PointTask::perform_mapping(
        MustEpochOp* must_epoch_owner /*=nullptr*/,
        const DeferMappingArgs* args /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      if (!map_all_regions(must_epoch_owner, args))
        return false;
      // Pul this on the stack to avoid querying it after we pass control
      // over to the slice owner
      const bool is_origin = is_origin_mapped();
      slice_owner->record_point_mapped(this, get_mapped_event());
      // Only return true if we're not origin-mapped, otherwise the slice
      // will end up taking care of us once it knows we're mapped
      return !is_origin;
    }

    //--------------------------------------------------------------------------
    void PointTask::perform_inlining(
        VariantImpl* variant, const std::deque<InstanceSet>& parent_regions)
    //--------------------------------------------------------------------------
    {
      SingleTask::perform_inlining(variant, parent_regions);
      slice_owner->record_point_mapped(this, get_mapped_event());
    }

    //--------------------------------------------------------------------------
    bool PointTask::finalize_map_task_output(
        Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
        MustEpochOp* must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      if (!SingleTask::finalize_map_task_output(
              input, output, must_epoch_owner))
        return false;
      // Once we know we're going to map this point task on this target
      // processor then we can get the concurrent analysis in flight
      // and record any map applied conditions for it, note we'll always have
      // a non-null args for concurrent tasks so points can map in parallel
      if (concurrent_task)
      {
        legion_assert(target_proc.exists());
        legion_assert(concurrent_postcondition.exists());
        concurrent_precondition.interpreted = Runtime::create_rt_user_event();
        if (must_epoch_task)
        {
          legion_assert(is_origin_mapped());
          must_epoch->rendezvous_concurrent_mapped(
              concurrent_precondition.interpreted);
        }
        else
          slice_owner->rendezvous_concurrent_mapped(
              index_point, target_proc, concurrent_color,
              concurrent_precondition.interpreted);
      }
      return true;
    }

    //--------------------------------------------------------------------------
    bool PointTask::replicate_task(void)
    //--------------------------------------------------------------------------
    {
      // Pull this onto the stack since it is unsafe to read it after we
      // call the base class method
      SliceTask* owner = slice_owner;
      const RtEvent event = get_mapped_event();
      const bool result = SingleTask::replicate_task();
      if (result)
        owner->record_point_mapped(this, event);
      return result;
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_future_size(
        size_t return_type_size, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(!elide_future_return);
      slice_owner->handle_future_size(return_type_size, index_point);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_output_extent(
        unsigned idx, const DomainPoint& color, const DomainPoint& extent)
    //--------------------------------------------------------------------------
    {
      slice_owner->record_output_extent(idx, color, extent);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_output_registered(
        RtEvent registered, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      slice_owner->record_output_registered(registered, applied_events);
    }

    //--------------------------------------------------------------------------
    void PointTask::shard_off(RtEvent mapped_precondition)
    //--------------------------------------------------------------------------
    {
      // Should only be happening for must-epoch operations
      legion_assert(must_epoch != nullptr);
      slice_owner->record_point_mapped(
          this, mapped_precondition, true /*shard off*/);
      SingleTask::shard_off(mapped_precondition);
    }

    //--------------------------------------------------------------------------
    bool PointTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_origin) && stealable);
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
    bool PointTask::is_output_grouped(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      return slice_owner->is_output_grouped(idx);
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind PointTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return POINT_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // Invalidate the logical context so child operations that still have
      // mapping references can begin committing
      slice_owner->record_point_complete(effects);
      if (execution_context != nullptr)
        execution_context->invalidate_logical_context();
      complete_operation(effects);
    }

    //--------------------------------------------------------------------------
    void PointTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> commit_preconditions;
      if (execution_context != nullptr)
      {
        slice_owner->return_privileges(execution_context, commit_preconditions);
        // Invalidate any context that we had so that the child
        // operations can begin committing
        std::set<RtEvent> point_preconditions;
        execution_context->invalidate_region_tree_contexts(
            false, commit_preconditions);
      }
      if (profiling_reported.exists())
      {
        finalize_single_task_profiling();
        commit_preconditions.insert(profiling_reported);
      }
      RtEvent commit_precondition;
      if (!commit_preconditions.empty())
        commit_precondition = Runtime::merge_events(commit_preconditions);
      // A little strange here, but we don't directly commit this
      // operation, instead we just tell our slice that we are commited
      // In the deactivation of the slice task is when we will actually
      // have our commit call done
      slice_owner->record_point_committed(commit_precondition);
    }

    //--------------------------------------------------------------------------
    bool PointTask::send_task(
        Processor target, std::vector<SingleTask*>& others)
    //--------------------------------------------------------------------------
    {
      SliceTask* owner = slice_owner;
      if (owner->send_task(target, this, others))
        owner->deactivate();
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::pack_task(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_single_task(rez, target);
      rez.serialize(orig_task);
      if (concurrent_task)
      {
        if (is_replaying())
          rez.serialize(concurrent_precondition.traced);
        else
          rez.serialize(concurrent_precondition.interpreted);
        rez.serialize(concurrent_postcondition);
      }
      // Return false since point tasks should always be deactivated
      // once they are sent to a remote node
      return false;
    }

    //--------------------------------------------------------------------------
    bool PointTask::unpack_task(
        Deserializer& derez, Processor current, AddressSpaceID,
        std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      // Get the context information from our slice owner
      parent_ctx = slice_owner->get_context();
      parent_task = parent_ctx->get_task();
      unpack_single_task(derez, ready_events);
      derez.deserialize(orig_task);
      if (concurrent_task)
      {
        if (is_replaying())
          derez.deserialize(concurrent_precondition.traced);
        else
          derez.deserialize(concurrent_precondition.interpreted);
        derez.deserialize(concurrent_postcondition);
      }
      set_current_proc(current);
      set_provenance(slice_owner->get_provenance(), false /*has ref*/);
      if (is_origin_mapped())
      {
        // We're not going to get a callback from the context if we're a leaf
        if (!is_leaf())
        {
          IndividualRemoteMapped rez;
          {
            RezCheck z2(rez);
            rez.serialize<SingleTask*>(orig_task);
            rez.serialize(get_mapped_event());
          }
          rez.dispatch(orig_proc.address_space());
        }
        else
          complete_mapping();
        slice_owner->record_point_mapped(this, get_mapped_event());
      }
      if (implicit_profiler != nullptr)
        implicit_profiler->register_operation(this);
      return false;
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_future(
        ApEvent effects, FutureInstance* instance, const void* metadata,
        size_t metasize, FutureFunctor* functor, Processor future_proc,
        bool own_functor)
    //--------------------------------------------------------------------------
    {
      if ((instance != nullptr) && (instance->size > 0) &&
          (shard_manager == nullptr))
        check_future_return_bounds(instance);
      slice_owner->handle_future(
          effects, index_point, instance, metadata, metasize, functor,
          future_proc, own_functor);
    }

    //--------------------------------------------------------------------------
    void PointTask::handle_mispredication(void)
    //--------------------------------------------------------------------------
    {
      // First thing: increment the meta-task counts since we decremented
      // them in case we didn't end up running
      runtime->increment_total_outstanding_tasks(
          MispredicationTaskArgs::TASK_ID, true /*meta*/);
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
      runtime->outstanding_counts[MispredicationTaskArgs::TASK_ID].fetch_add(1);
#endif
      slice_owner->set_predicate_false_result(index_point, execution_context);
      // Pretend like we executed the task
      execution_context->handle_mispredication();
    }

    //--------------------------------------------------------------------------
    uint64_t PointTask::order_collectively_mapped_unbounded_pools(
        uint64_t lamport_clock, bool need_result)
    //--------------------------------------------------------------------------
    {
      if (must_epoch_task)
      {
        legion_assert(is_origin_mapped());
        return must_epoch->collective_lamport_allreduce(
            lamport_clock, need_result);
      }
      else
        return slice_owner->collective_lamport_allreduce(
            lamport_clock, need_result);
    }

    //--------------------------------------------------------------------------
    ApEvent PointTask::order_concurrent_launch(ApEvent start, VariantImpl* impl)
    //--------------------------------------------------------------------------
    {
      legion_assert(target_processors.size() == 1);
      legion_assert(concurrent_postcondition.exists());
      // To order concurrent launches we do a two-phase algorithm here
      // 1. We do a barrier across all the point tasks to make sure that
      //    the preconditions for all the point tasks are ready before we
      //    start the lamport clock protocol. In the interpreted case this
      //    barrier is done with a butterfly network of Realm events. In the
      //    tracing case we do this with a proper Realm barrier since we
      //    can assume it will be done many times. The concurrent_postcondition
      //    event represents the event for the end of this barrier
      // 2. We do a max all-reduce of lamport clocks from each of the point
      //    tasks so that we can establish an ordering of concurrent index
      //    space task launches that use overlapping processors.
      const OrderConcurrentLaunchArgs args(
          this, target_processors.front(), start,
          (impl->is_concurrent() || must_epoch_task) ? selected_variant : 0);
      RtEvent precondition;
      if (start.exists())
        precondition = Runtime::protect_event(start);
      if (is_replaying())
        runtime->phase_barrier_arrive(
            concurrent_precondition.traced, 1 /*count*/, precondition);
      else
        Runtime::trigger_event(
            concurrent_precondition.interpreted, precondition);
      // Give this very high priority as it is likely on the critical path
      runtime->issue_runtime_meta_task(
          args, LG_RESOURCE_PRIORITY, concurrent_postcondition);
      return args.ready;
    }

    //--------------------------------------------------------------------------
    void PointTask::concurrent_allreduce(
        ProcessorManager* manager, uint64_t lamport_clock, VariantID vid,
        bool poisoned)
    //--------------------------------------------------------------------------
    {
      slice_owner->concurrent_allreduce(
          this, manager, lamport_clock, vid, poisoned);
    }

    //--------------------------------------------------------------------------
    bool PointTask::check_concurrent_variant(VariantID vid)
    //--------------------------------------------------------------------------
    {
      if (vid == 0)
      {
        VariantImpl* impl =
            runtime->find_variant_impl(task_id, selected_variant);
        if (!impl->is_concurrent())
          return true;
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Mapper " << *mapper << " selected a concurrent variant "
              << selected_variant << " for point task " << *this
              << " of a concurrent task launch but selected a "
              << "non-concurrent variant for a different point task. All point "
              << "tasks in a concurrent index task launch must be the same if "
              << "any of them are going to be a concurrent variant.";
        error.raise();
      }
      else if (vid != selected_variant)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Mapper " << *mapper << " selected a concurrent variant "
              << selected_variant << " for point task " << *this
              << " of a concurrent task launch but selected a different "
              << "concurrent variant " << vid << " for a different point task. "
              << "All point tasks in a concurrent index task launch must use "
              << "the same concurrent task variant.";
        error.raise();
      }
      return true;
    }

    //--------------------------------------------------------------------------
    void PointTask::perform_concurrent_task_barrier(void)
    //--------------------------------------------------------------------------
    {
      // Check that this is a concurrent index space task launch
      if (!is_concurrent())
      {
        Error error(LEGION_INTERFACE_EXCEPTION);
        error << "Illegal concurrent task barrier in task " << *this
              << " which is not part of a concurrent index space task. "
              << "Concurrent task barriers are only permitted in concurrent "
              << "index space tasks.";
        error.raise();
      }
      if (!concurrent_task_barrier.exists())
      {
        concurrent_task_barrier =
            slice_owner->get_concurrent_task_barrier(concurrent_color);
        if (!concurrent_task_barrier.exists())
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error
              << "Illegal concurrent task barrier in task " << *this
              << " which is not a task variant that requested support for "
              << "concurrent barriers. To request support you must mark the "
              << "task variant as needing 'concurrent_barrier' support in the "
              << "task variant registrar.";
          error.raise();
        }
      }
      runtime->phase_barrier_arrive(concurrent_task_barrier, 1 /*count*/);
      concurrent_task_barrier.wait();
      Runtime::advance_barrier(concurrent_task_barrier);
      // If you ever fail this assertion then we exhausted the number
      // of generations in a barrier. Hopefully CUDA will fix its bug
      // before we ever need to deal with this
      legion_assert(concurrent_task_barrier.exists());
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
      legion_assert(idx < get_region_count());
      RegionRequirement& req = logical_regions[idx];
      legion_assert(req.handle_type != LEGION_SINGULAR_PROJECTION);
      req.region = result;
      req.handle_type = LEGION_SINGULAR_PROJECTION;
      // Check to see if the region is a NO_REGION,
      // if it is then switch the privilege to NO_ACCESS
      if (req.region == LogicalRegion::NO_REGION)
        req.privilege = LEGION_NO_ACCESS;
    }

    //--------------------------------------------------------------------------
    void PointTask::initialize_point(
        SliceTask* owner, const DomainPoint& point,
        const FutureMap& point_arguments, bool inline_task,
        const std::vector<FutureMap>& point_futures,
        bool record_future_pointwise_dependences)
    //--------------------------------------------------------------------------
    {
      slice_owner = owner;
      // Get our point
      index_point = point;
      // Get our argument
      if (point_arguments.impl != nullptr)
      {
        Future f = point_arguments.impl->get_future(point, true /*internal*/);
        if (f.impl != nullptr)
        {
          if (inline_task)
          {
            local_args = f.impl->get_buffer(
                runtime->runtime_system_memory, &local_arglen);
          }
          else
          {
            // Request the local buffer
            f.impl->request_runtime_instance(this);
            // Make sure that it is ready
            const RtEvent ready = f.impl->subscribe();
            if (ready.exists() && !ready.has_triggered())
              ready.wait();
            local_args = f.impl->find_runtime_buffer(parent_ctx, local_arglen);
          }
        }
      }
      if (!point_futures.empty())
      {
        const int context_depth = parent_ctx->get_depth();
        for (const FutureMap& future_map : point_futures)
        {
          if (!future_map.impl->future_map_domain->contains_point(point))
          {
            this->futures.emplace_back(Future());
            continue;
          }
          this->futures.emplace_back(
              future_map.impl->get_future(point, true /*internal*/));
          if (record_future_pointwise_dependences)
          {
            const RtEvent pre = future_map.impl->find_pointwise_dependence(
                point, context_depth);
            if (pre.exists() && !std::binary_search(
                                    pointwise_mapping_dependences.begin(),
                                    pointwise_mapping_dependences.end(), pre))
            {
              pointwise_mapping_dependences.emplace_back(pre);
              std::sort(
                  pointwise_mapping_dependences.begin(),
                  pointwise_mapping_dependences.end());
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent PointTask::perform_pointwise_analysis(void)
    //--------------------------------------------------------------------------
    {
      return Runtime::merge_events(pointwise_mapping_dependences);
    }

    //--------------------------------------------------------------------------
    void PointTask::complete_replay(ApEvent instance_ready_event)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_leaf());
      legion_assert(is_origin_mapped());
      legion_assert(!target_processors.empty());
      legion_assert(single_task_termination.exists());
      legion_assert(region_preconditions.empty());
      const AddressSpaceID target_space =
          target_processors.front().address_space();
      // Check to see if we're replaying this locally or remotely
      if (target_space != runtime->address_space)
      {
        // This is the remote case, pack it up and ship it over
        // Update our target_proc so that the sending code is correct
        RemoteTaskReplay rez;
        {
          RezCheck z(rez);
          rez.serialize(instance_ready_event);
          rez.serialize(target_processors.front());
          rez.serialize(SLICE_TASK_KIND);
          slice_owner->pack_task(rez, target_space);
        }
        rez.dispatch(target_space);
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
    bool PointTask::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      return slice_owner->find_shard_participants(shards);
    }

    //--------------------------------------------------------------------------
    RtEvent PointTask::convert_collective_views(
        unsigned requirement_index, unsigned analysis_index,
        LogicalRegion region, const InstanceSet& targets,
        InnerContext* physical_ctx, CollectiveMapping*& analysis_mapping,
        bool& first_local,
        op::vector<op::FieldMaskMap<InstanceView> >& target_views,
        std::map<InstanceView*, size_t>& collective_arrivals)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_collective_rendezvous(
          unique_op_id, requirement_index, analysis_index);
      return slice_owner->convert_collective_views(
          requirement_index, analysis_index, region, targets, physical_ctx,
          analysis_mapping, first_local, target_views, collective_arrivals);
    }

    //--------------------------------------------------------------------------
    RtEvent PointTask::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return slice_owner->perform_collective_versioning_analysis(
          index, handle, tracker, mask, parent_req_index);
    }

    //--------------------------------------------------------------------------
    void PointTask::perform_replicate_collective_versioning(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, CollectiveVersioningBase::RegionVersioning>&
            to_perform)
    //--------------------------------------------------------------------------
    {
      legion_assert(shard_manager != nullptr);
      if (IS_COLLECTIVE(regions[index]) ||
          std::binary_search(
              check_collective_regions.begin(), check_collective_regions.end(),
              index))
        slice_owner->perform_replicate_collective_versioning(
            index, parent_req_index, to_perform);
      else
        SingleTask::perform_replicate_collective_versioning(
            index, parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    void PointTask::convert_replicate_collective_views(
        const CollectiveViewCreatorBase::RendezvousKey& key,
        std::map<
            LogicalRegion, CollectiveViewCreatorBase::CollectiveRendezvous>&
            rendezvous)
    //--------------------------------------------------------------------------
    {
      legion_assert(shard_manager != nullptr);
      if (IS_COLLECTIVE(regions[key.region_index]) ||
          std::binary_search(
              check_collective_regions.begin(), check_collective_regions.end(),
              key.region_index))
        slice_owner->convert_replicate_collective_views(key, rendezvous);
      else
        SingleTask::convert_replicate_collective_views(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    void PointTask::record_intra_space_dependences(
        unsigned index, const std::vector<DomainPoint>& dependences)
    //--------------------------------------------------------------------------
    {
      // Should have been caught by the caller
      legion_assert(!dependences.empty());
      if (dependences.size() == 1)
      {
        legion_assert(dependences.back() == index_point);
        return;
      }
      if (concurrent_task)
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Concurrent index space task " << *this
              << " has intra-index-space dependences on region requirement "
              << index
              << ". It is illegal to have intra-index-space dependences "
              << "on concurrent executions because the resulting execution is "
              << "guaranteed to hang.";
        error.raise();
      }
      if (!check_collective_regions.empty())
      {
        if (mapper == nullptr)
          mapper = runtime->find_mapper(current_proc, map_id);
        Error error(LEGION_MAPPER_EXCEPTION);
        error
            << "Mapper " << *mapper << " asked for collective region "
            << "checks for index task " << *this << " but this task has "
            << "intra-index-space task dependences. Collective behavior "
            << "cannot be analyzed on task with inter-index-space dependences.";
        error.raise();
      }
      // Scan through the list until we find ourself
      for (unsigned idx = 0; idx < dependences.size(); idx++)
      {
        if (dependences[idx] == index_point)
        {
          RegionRequirement& req = logical_regions[index];
          // If we've got a prior dependence then record it
          if (idx > 0)
          {
            const DomainPoint& prev = dependences[idx - 1];
            const RtEvent pre = slice_owner->find_intra_space_dependence(prev);
            if (!std::binary_search(
                    pointwise_mapping_dependences.begin(),
                    pointwise_mapping_dependences.end(), pre))
            {
              pointwise_mapping_dependences.emplace_back(pre);
              std::sort(
                  pointwise_mapping_dependences.begin(),
                  pointwise_mapping_dependences.end());
            }
            // If we're not the first point in line for this region requirement
            // then we shouldn't have a discard qualifier
            if (IS_WRITE_DISCARD(req))
              req.privilege &= ~LEGION_DISCARD_INPUT_MASK;
            // We know we only need a dependence on the previous point but
            // Legion Spy is stupid, so log everything we have a
            // precondition on even if it is transitively implied
            for (unsigned idx2 = 0; idx2 < idx; idx2++)
              LegionSpy::log_intra_space_dependence(
                  unique_op_id, dependences[idx2]);
          }
          // If we're not the last point in line for this region requirement
          // then we shouldn't have an output discard qualifier
          else if (IS_OUTPUT_DISCARD(req) && (dependences.size() > 1))
            req.privilege &= ~LEGION_DISCARD_OUTPUT_MASK;
          return;
        }
      }
      // We should never get here
      std::abort();
    }

    //--------------------------------------------------------------------------
    void PointTask::record_pointwise_dependence(
        uint64_t previous_context_index, const DomainPoint& previous_point,
        ShardID shard)
    //--------------------------------------------------------------------------
    {
      const RtEvent pre = parent_ctx->find_pointwise_dependence(
          previous_context_index, previous_point, shard);
      if (pre.exists())
      {
        pointwise_mapping_dependences.emplace_back(pre);
        std::sort(
            pointwise_mapping_dependences.begin(),
            pointwise_mapping_dependences.end());
      }
    }

    //--------------------------------------------------------------------------
    bool PointTask::has_remaining_inlining_dependences(
        std::map<PointTask*, unsigned>& remaining,
        std::map<RtEvent, std::vector<PointTask*> >& event_deps) const
    //--------------------------------------------------------------------------
    {
      if (pointwise_mapping_dependences.empty())
        return false;
      unsigned count = 0;
      for (const RtEvent& event : pointwise_mapping_dependences)
      {
        if (event.has_triggered())
          continue;
        count++;
        event_deps[event].emplace_back(const_cast<PointTask*>(this));
      }
      if (count > 0)
      {
        remaining[const_cast<PointTask*>(this)] = count;
        return true;
      }
      else
        return false;
    }

    //--------------------------------------------------------------------------
    void PointTask::complete_point_projection(void)
    //--------------------------------------------------------------------------
    {
      update_no_access_regions();
      // Log our requirements that we computed
      UniqueID our_uid = get_unique_id();
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
        log_requirement(our_uid, idx, logical_regions[idx]);
    }

  }  // namespace Internal
}  // namespace Legion
