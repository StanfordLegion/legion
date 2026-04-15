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

#include "legion/tasks/slice.h"
#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"
#include "legion/managers/processor.h"
#include "legion/operations/mustepoch.h"
#include "legion/tasks/index.h"
#include "legion/tasks/point.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Slice Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SliceTask::SliceTask(void) : MultiTask()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    SliceTask::~SliceTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void SliceTask::activate(void)
    //--------------------------------------------------------------------------
    {
      MultiTask::activate();
      num_unmapped_points = 0;
      num_uncompleted_points.store(0);
      num_uncommitted_points = 0;
      index_owner = nullptr;
      remote_unique_id = get_unique_id();
      origin_mapped = false;
      // Slice tasks always already have their options selected
      options_selected = true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Deactivate all our points
      for (PointTask* const pt : points) pt->deactivate();
      points.clear();
      legion_assert(local_regions.empty());
      legion_assert(local_fields.empty());
      commit_preconditions.clear();
      created_regions.clear();
      created_fields.clear();
      created_field_spaces.clear();
      created_index_spaces.clear();
      created_index_partitions.clear();
      MultiTask::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    Operation* SliceTask::get_origin_operation(void)
    //--------------------------------------------------------------------------
    {
      return index_owner;
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void SliceTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
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
      legion_assert(!points.empty());
      if (points.size() == 1)
        return;
      const AddressSpaceID target_space =
          points[0]->target_proc.address_space();
      for (unsigned idx = 1; idx < points.size(); idx++)
      {
        if (target_space != points[idx]->target_proc.address_space())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output: two different points in one "
                << "slice of " << *this << " mapped to processors in two "
                << "different address spaces (" << target_space << " and "
                << points[idx]->target_proc.address_space()
                << ") which is illegal.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::update_target_processor(void)
    //--------------------------------------------------------------------------
    {
      if (points.empty())
        return;
      if (runtime->safe_mapper)
        check_target_processors();
      this->target_proc = points[0]->target_proc;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      update_target_processor();
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
          // The runtime will deactivate this task
          // after it has been sent
          return false;
        }
        return true;
      }
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
    bool SliceTask::perform_mapping(
        MustEpochOp* epoch_owner /*=nullptr*/,
        const DeferMappingArgs* args /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      // Should never get duplicate invocations here
      legion_assert(args == nullptr);
      // Check to see if we already enumerated all the points, if
      // not then do so now
      const bool make_points = points.empty();
      if (make_points)
        enumerate_points(false /*inlining*/);
      // Enqueue all the point tasks as ready to map
      // Make a copy of the points data structure because as soon as we
      // kick off a task it might come back and start mutating the
      // points data structure out from under us and even cleaning up
      // the slice task object before we're done
      const std::vector<PointTask*> copy(points.begin(), points.end());
      for (PointTask* const point : copy)
      {
        // If we just made this point then perform the pointwise analysis
        // on it before we can go about trying to map it
        RtEvent point_precondition;
        if (make_points)
          point_precondition = point->perform_pointwise_analysis();
        if (is_origin_mapped())
        {
          // We can start the mapping for this point task now
          TriggerTaskArgs trigger_args(point, parent_ctx->did);
          runtime->issue_runtime_meta_task(
              trigger_args, LG_THROUGHPUT_WORK_PRIORITY, point_precondition);
        }
        else
          point->enqueue_ready_task(!make_points, point_precondition);
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void SliceTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      legion_assert(!points.empty());
      // Launch all of our child points
      for (PointTask* const pt : points) pt->launch_task(inline_task);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_stealable(void) const
    //--------------------------------------------------------------------------
    {
      return ((!map_origin) && stealable);
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_output_global(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < output_region_options.size());
      return output_region_options[idx].global_indexing();
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_output_grouped(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < output_region_options.size());
      return output_region_options[idx].grouped_fields();
    }

    //--------------------------------------------------------------------------
    bool SliceTask::is_output_valid(unsigned idx) const
    //--------------------------------------------------------------------------
    {
      legion_assert(idx < output_region_options.size());
      return output_region_options[idx].valid_requirement();
    }

    //--------------------------------------------------------------------------
    TaskOp::TaskKind SliceTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return SLICE_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    bool SliceTask::send_task(
        Processor target, PointTask* point, std::vector<SingleTask*>& others)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_origin_mapped());
      legion_assert(std::is_sorted(others.begin(), others.end()));
      std::vector<PointTask*> to_send(1, point);
      for (PointTask* const pt : points)
      {
        if (pt == point)
          continue;
        std::vector<SingleTask*>::iterator finder =
            std::lower_bound(others.begin(), others.end(), pt);
        if ((finder != others.end()) && (*finder == pt))
        {
          to_send.emplace_back(pt);
          others.erase(finder);
        }
      }
      legion_assert(to_send.size() <= points.size());
      bool trigger_mapped = false;
      bool trigger_children_commit = false;
      std::vector<Color> concurrent_colors;
      if (to_send.size() == points.size())
      {
        TaskMessage rez;
        bool deactivate;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(SLICE_TASK_KIND);
          deactivate = pack_task(rez, target.address_space());
        }
        rez.dispatch(target.address_space());
        return deactivate;
      }
      else
      {
        // This is the nasty case where we need to pack this slice and
        // then only send a subset of the points to the remote node
        TaskMessage rez;
        {
          RezCheck z(rez);
          rez.serialize(target);
          rez.serialize(SLICE_TASK_KIND);
          pack_slice_task(rez, target.address_space(), to_send);
        }
        rez.dispatch(target.address_space());
        // Now take the lock and remove the sent points and see if
        // we need to trigger anything downstream because we won't
        // have these points completing anymore
        std::sort(to_send.begin(), to_send.end());
        AutoLock o_lock(op_lock);
        for (std::vector<PointTask*>::iterator it = points.begin();
             it != points.end();
             /*nothing*/)
        {
          if (std::binary_search(to_send.begin(), to_send.end(), *it))
            it = points.erase(it);
          else
            it++;
        }
        if (concurrent_task)
        {
          if (is_remote() && (concurrent_points == points.size()))
            send_rendezvous_concurrent_mapped();
          // Decrement the group point counts of the points that
          // were sent away
          ConcurrentColoringFunctor* functor =
              runtime->find_concurrent_coloring_functor(concurrent_functor);
          for (PointTask* const pt : to_send)
          {
            Color color = functor->color(pt->index_point, index_domain);
            std::map<Color, ConcurrentGroup>::iterator finder =
                concurrent_groups.find(color);
            legion_assert(finder != concurrent_groups.end());
            legion_assert(finder->second.group_points > 0);
            legion_assert(
                finder->second.point_tasks.size() <
                finder->second.group_points);
            finder->second.group_points--;
            // See if we have any concurrent mapping to trigger
            if ((finder->second.group_points > 0) &&
                (finder->second.point_tasks.size() ==
                 finder->second.group_points))
              concurrent_colors.emplace_back(color);
          }
        }
        legion_assert(to_send.size() <= num_unmapped_points);
        legion_assert(to_send.size() <= num_uncommitted_points);
        num_unmapped_points -= to_send.size();
        trigger_mapped = (num_unmapped_points == 0);
        num_uncommitted_points -= to_send.size();
        trigger_children_commit = (num_uncommitted_points == 0);
      }
      for (const Color& color : concurrent_colors)
      {
        std::map<Color, ConcurrentGroup>::iterator finder =
            concurrent_groups.find(color);
        legion_assert(finder != concurrent_groups.end());
        if (is_remote())
        {
          SliceConcurrentRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(index_owner);
            rez.serialize(this);
            rez.serialize(finder->first);
            rez.serialize(finder->second.group_points);
            rez.serialize(finder->second.lamport_clock);
            rez.serialize(finder->second.variant);
            rez.serialize(finder->second.poisoned);
          }
          rez.dispatch(orig_proc.address_space());
        }
        else
          index_owner->concurrent_allreduce(
              finder->first, this, runtime->address_space,
              finder->second.group_points, finder->second.lamport_clock,
              finder->second.variant, finder->second.poisoned);
      }
      if (trigger_mapped)
        complete_mapping();
      const unsigned remaining =
          num_uncompleted_points.fetch_sub(to_send.size());
      legion_assert(to_send.size() <= remaining);
      if (remaining == to_send.size())
        complete_execution();
      if (trigger_children_commit)
        trigger_children_committed();
      // Deactivate the points that we removed and sent
      for (PointTask* const pt : to_send) pt->deactivate();
      return false;
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_slice_task(
        Serializer& rez, AddressSpaceID target,
        const std::vector<PointTask*>& points_to_send)
    //--------------------------------------------------------------------------
    {
      // Check to see if we are stealable or not yet fully sliced,
      // if both are false and we're not remote, then we can send the state
      // now or check to see if we are remotely mapped
      RezCheck z(rez);
      // Preamble used in TaskOp::unpack
      rez.serialize(points_to_send.size());
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
          legion_assert(future_map.impl != nullptr);
          future_map.impl->pack_future_map(rez, target);
        }
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
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        provenance->serialize(rez);
      else
        Provenance::serialize_null(rez);
      for (PointTask* const pt : points_to_send) pt->pack_task(rez, target);
      // If we don't have any points, we have to pack up the argument map
      // and any trace info that we need for doing remote tracing
      if (points_to_send.empty())
      {
        if (point_arguments.impl != nullptr)
          point_arguments.impl->pack_future_map(rez, target);
        else
          rez.serialize<DistributedID>(0);
        rez.serialize<size_t>(point_futures.size());
        for (const FutureMap& future_map : point_futures)
        {
          FutureMapImpl* impl = future_map.impl;
          impl->pack_future_map(rez, target);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool SliceTask::pack_task(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      pack_slice_task(rez, target, points);
      if (is_origin_mapped() && !is_remote())
      {
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
    bool SliceTask::unpack_task(
        Deserializer& derez, Processor current, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t num_points;
      derez.deserialize(num_points);
      unpack_multi_task(derez, ready_events);
      set_current_proc(current);
      derez.deserialize(index_owner);
      derez.deserialize(remote_unique_id);
      derez.deserialize(origin_mapped);
      parent_ctx = InnerContext::unpack_inner_context(derez);
      derez.deserialize(internal_space);
      LegionSpy::log_slice_slice(remote_unique_id, get_unique_id());
      if (implicit_profiler != nullptr)
        implicit_profiler->register_slice_owner(
            remote_unique_id, get_unique_op_id());
      num_unmapped_points = num_points;
      num_uncompleted_points.store(num_points);
      num_uncommitted_points = num_points;
      if (!elide_future_return)
      {
        if (redop == 0)
          future_map = FutureMapImpl::unpack_future_map(derez, parent_ctx);
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
      // Unpack the provenance before unpacking any point tasks so
      // that they can pick it up as well
      set_provenance(Provenance::deserialize(derez), true /*has ref*/);
      for (unsigned idx = 0; idx < num_points; idx++)
      {
        PointTask* point = runtime->get_operation<PointTask>();
        point->slice_owner = this;
        point->unpack_task(derez, current, ready_events);
        point->parent_ctx = parent_ctx;
        points.emplace_back(point);
        LegionSpy::log_slice_point(
            get_unique_id(), point->get_unique_id(), point->index_point);
      }
      if (concurrent_task)
      {
        // Update the concurrent groups based on the points
        ConcurrentColoringFunctor* functor =
            runtime->find_concurrent_coloring_functor(concurrent_functor);
        for (PointTask* const pt : points)
        {
          Color color = functor->color(pt->index_point, index_domain);
          std::map<Color, ConcurrentGroup>::iterator finder =
              concurrent_groups.find(color);
          legion_assert(finder != concurrent_groups.end());
          finder->second.group_points++;
        }
      }
      if (num_points == 0)
      {
        point_arguments = FutureMapImpl::unpack_future_map(derez, parent_ctx);
        size_t num_point_futures;
        derez.deserialize(num_point_futures);
        if (num_point_futures > 0)
        {
          point_futures.resize(num_point_futures);
          for (unsigned idx = 0; idx < num_point_futures; idx++)
            point_futures[idx] =
                FutureMapImpl::unpack_future_map(derez, parent_ctx);
        }
      }
      if (implicit_profiler != nullptr)
        implicit_profiler->register_operation(this);
      // Return true to add this to the ready queue
      return true;
    }

    //--------------------------------------------------------------------------
    void SliceTask::perform_inlining(
        VariantImpl* variant, const std::deque<InstanceSet>& parent_instances)
    //--------------------------------------------------------------------------
    {
      // Need to handle inter-space dependences correctly here
      std::map<PointTask*, unsigned> remaining;
      std::map<RtEvent, std::vector<PointTask*>> event_deps;
      for (PointTask* const pt : points)
        if (!pt->has_remaining_inlining_dependences(remaining, event_deps))
          pt->perform_inlining(variant, parent_instances);
      while (!remaining.empty())
      {
        [[maybe_unused]] bool found =
            false;  // should find at least one each iteration
        for (std::map<PointTask*, unsigned>::iterator it = remaining.begin();
             it != remaining.end();
             /*nothing*/)
        {
          if (it->second == 0)
          {
            const RtEvent mapped = it->first->get_mapped_event();
            it->first->perform_inlining(variant, parent_instances);
            found = true;
            legion_assert(mapped.has_triggered());
            std::map<RtEvent, std::vector<PointTask*>>::const_iterator finder =
                event_deps.find(mapped);
            if (finder != event_deps.end())
            {
              for (unsigned idx = 0; idx < finder->second.size(); idx++)
              {
                std::map<PointTask*, unsigned>::iterator point_finder =
                    remaining.find(finder->second[idx]);
                legion_assert(point_finder != remaining.end());
                legion_assert(point_finder->second > 0);
                point_finder->second--;
              }
              event_deps.erase(finder);
            }
            std::map<PointTask*, unsigned>::iterator to_delete = it++;
            remaining.erase(to_delete);
          }
          else
            it++;
        }
        legion_assert(found);
      }
    }

    //--------------------------------------------------------------------------
    SliceTask* SliceTask::clone_as_slice_task(
        IndexSpace is, Processor p, bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      SliceTask* result = runtime->get_operation<SliceTask>();
      result->initialize_base_task(
          parent_ctx, Predicate::TRUE_PRED, this->task_id, get_provenance());
      result->clone_multi_from(this, is, p, recurse, stealable);
      result->index_owner = this->index_owner;
      LegionSpy::log_slice_slice(get_unique_id(), result->get_unique_id());
      if (implicit_profiler != nullptr)
        implicit_profiler->register_slice_owner(
            get_unique_op_id(), result->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void SliceTask::reduce_future(
        const DomainPoint& point, FutureInstance* inst, ApEvent effects)
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
          legion_assert(
              temporary_futures.find(point) == temporary_futures.end());
          temporary_futures[point] = std::make_pair(inst, effects);
        }
        else
        {
          // If we're not doing serdez functions, we'll grab the first
          // one of these instances as the target for us to reduce into
          if ((serdez_redop_fns == nullptr) && (reduction_instance == nullptr))
          {
            AutoLock o_lock(op_lock);
            // See if we lost the race
            if (reduction_instance == nullptr)
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
                reduction_instance = FutureInstance::create_local(
                    &reduction_op->identity, reduction_op->sizeof_rhs,
                    false /*own*/);
            }
          }
          if (!fold_reduction_future(inst, effects))
          {
            // save it to delete later
            AutoLock o_lock(op_lock);
            legion_assert(
                temporary_futures.find(point) == temporary_futures.end());
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
    void SliceTask::handle_future(
        ApEvent effects, const DomainPoint& point, FutureInstance* instance,
        const void* metadata, size_t metasize, FutureFunctor* functor,
        Processor future_proc, bool own_functor)
    //--------------------------------------------------------------------------
    {
      if (elide_future_return)
      {
        if (functor != nullptr)
        {
          legion_assert(instance == nullptr);
          legion_assert(metadata == nullptr);
          functor->callback_release_future();
          if (own_functor)
            delete functor;
        }
        else if ((instance != nullptr) && !instance->defer_deletion(effects))
          delete instance;
      }
      else if (redop > 0)
      {
        legion_assert(functor == nullptr);
        legion_assert(instance != nullptr);
        reduce_future(point, instance, effects);
        if (metadata != nullptr)
        {
          AutoLock o_lock(op_lock);
          if (reduction_metadata == nullptr)
          {
            reduction_metasize = metasize;
            reduction_metadata = malloc(metasize);
            memcpy(reduction_metadata, metadata, metasize);
          }
        }
      }
      else
      {
        legion_assert(future_handles != nullptr);
        std::map<DomainPoint, DistributedID>::const_iterator finder =
            future_handles->handles.find(point);
        legion_assert(finder != future_handles->handles.end());
        const ContextCoordinate coordinate(future_map_coordinate, point);
        RtEvent registered;
        FutureImpl* impl = runtime->find_or_create_future(
            finder->second, parent_ctx->did, coordinate, get_provenance(),
            false /*has global reference*/, registered);
        if (functor != nullptr)
        {
          legion_assert(instance == nullptr);
          legion_assert(metadata == nullptr);
          impl->set_result(effects, functor, own_functor, future_proc);
        }
        else
          impl->set_result(effects, instance, metadata, metasize);
        if (registered.exists())
        {
          AutoLock o_lock(op_lock);
          commit_preconditions.insert(registered);
        }
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::register_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch != nullptr);
      if (points.empty())
        enumerate_points(false /*inling*/);
      must_epoch->register_slice_task(this);
      for (PointTask* const point : points)
      {
        must_epoch->register_single_task(point, must_epoch_index);
      }
    }

    //--------------------------------------------------------------------------
    PointTask* SliceTask::clone_as_point_task(
        const DomainPoint& point, bool inline_task)
    //--------------------------------------------------------------------------
    {
      PointTask* result = runtime->get_operation<PointTask>();
      result->initialize_base_task(
          parent_ctx, Predicate::TRUE_PRED, this->task_id, get_provenance());
      result->clone_task_op_from(this, this->target_proc, false /*stealable*/);
      result->is_index_space = true;
      result->must_epoch_task = this->must_epoch_task;
      result->index_domain = this->index_domain;
      result->version_infos.resize(logical_regions.size());
      // Now figure out our local point information
      result->initialize_point(
          this, point, point_arguments, inline_task, point_futures,
          is_pointwise_analyzable());
      if (concurrent_task)
      {
        // Find the color for this point task
        ConcurrentColoringFunctor* functor =
            runtime->find_concurrent_coloring_functor(concurrent_functor);
        result->concurrent_color = functor->color(point, index_domain);
        if (is_replaying())
        {
          std::map<Color, ConcurrentGroup>::const_iterator finder =
              concurrent_groups.find(result->concurrent_color);
          legion_assert(finder != concurrent_groups.end());
          result->concurrent_precondition.traced =
              finder->second.precondition.traced;
          result->concurrent_postcondition = finder->second.precondition.traced;
        }
        else
        {
          std::map<Color, ConcurrentGroup>::const_iterator finder =
              concurrent_groups.find(result->concurrent_color);
          legion_assert(finder != concurrent_groups.end());
          result->concurrent_postcondition =
              finder->second.precondition.interpreted;
        }
      }
      LegionSpy::log_slice_point(
          get_unique_id(), result->get_unique_id(), result->index_point);
      return result;
    }

    //--------------------------------------------------------------------------
    size_t SliceTask::enumerate_points(bool inline_task)
    //--------------------------------------------------------------------------
    {
      Domain internal_domain;
      runtime->find_domain(internal_space, internal_domain);
      const size_t num_points = internal_domain.get_volume();
      legion_assert(num_points > 0);
      unsigned point_idx = 0;
      points.resize(num_points);
      // Enumerate all the points in our slice and make point tasks
      for (Domain::DomainPointIterator itr(internal_domain); itr;
           itr++, point_idx++)
        points[point_idx] = clone_as_point_task(itr.p, inline_task);
      // Compute any projection region requirements
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
      {
        const RegionRequirement& req = logical_regions[idx];
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
          continue;
        ProjectionFunction* function =
            runtime->find_projection_function(req.projection);
        std::map<unsigned, std::vector<PointwiseDependence>>::const_iterator
            finder = pointwise_dependences.find(idx);
        function->project_points(
            req, idx, index_domain, points,
            (finder == pointwise_dependences.end()) ? nullptr : &finder->second,
            parent_ctx->get_total_shards(), is_replaying());
      }
      // Update the no access regions
      for (PointTask* const pt : points) pt->complete_point_projection();
      if (concurrent_task)
      {
        // Set the counts back to zero for all the groups and then
        // count how many local points we're going to be expecting here
        for (std::pair<const Color, ConcurrentGroup>& group : concurrent_groups)
          group.second.group_points = 0;
        ConcurrentColoringFunctor* functor =
            runtime->find_concurrent_coloring_functor(concurrent_functor);
        for (PointTask* const pt : points)
        {
          Color color = functor->color(pt->index_point, index_domain);
          std::map<Color, ConcurrentGroup>::iterator finder =
              concurrent_groups.find(color);
          legion_assert(finder != concurrent_groups.end());
          finder->second.group_points++;
        }
      }
      // Mark how many points we have
      num_unmapped_points = num_points;
      num_uncompleted_points.store(num_points);
      num_uncommitted_points = num_points;
      return num_points;
    }

    //--------------------------------------------------------------------------
    void SliceTask::set_predicate_false_result(
        const DomainPoint& point, TaskContext* execution_context)
    //--------------------------------------------------------------------------
    {
      if (elide_future_return || (redop > 0))
        return;
      legion_assert(future_handles != nullptr);
      std::map<DomainPoint, DistributedID>::const_iterator finder =
          future_handles->handles.find(point);
      legion_assert(finder != future_handles->handles.end());
      const ContextCoordinate coordinate(future_map_coordinate, point);
      RtEvent registered;
      FutureImpl* impl = runtime->find_or_create_future(
          finder->second, parent_ctx->did, coordinate, get_provenance(),
          false /*has global reference*/, registered);
      if (predicate_false_future.impl == nullptr)
      {
        if (predicate_false_result.get_size() > 0)
          impl->set_local(
              predicate_false_result.get_buffer(),
              predicate_false_result.get_size(), false /*own*/);
        else
          impl->set_result(ApEvent::NO_AP_EVENT, nullptr);
      }
      else
        impl->set_result(execution_context, predicate_false_future.impl);
      if (registered.exists())
      {
        AutoLock o_lock(op_lock);
        commit_preconditions.insert(registered);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // For remote cases we have to keep track of the events for
      // returning any created logical state, we can't commit until
      // it is returned or we might prematurely release the references
      // that we hold on the version state objects
      if (is_remote())
      {
        // Send back the message saying that this slice is complete
        SliceRemoteComplete rez;
        pack_remote_complete(rez, effects);
        rez.dispatch(orig_proc.address_space());
      }
      else
      {
        legion_assert(temporary_futures.empty());
        legion_assert(reduction_instance == nullptr);
        legion_assert(serdez_redop_state == nullptr);
        index_owner->return_slice_complete(
            points.size(), effects, reduction_metadata, reduction_metasize);
        // No longer own the buffer so clear it
        reduction_metadata = nullptr;
      }
      complete_operation();
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      RtEvent commit_precondition;
      if (!commit_preconditions.empty())
        commit_precondition = Runtime::merge_events(commit_preconditions);
      if (is_remote())
      {
        SliceRemoteCommit rez;
        pack_remote_commit(rez, commit_precondition);
        rez.dispatch(orig_proc.address_space());
      }
      else
      {
        // created and deleted privilege information already passed back
        // futures already sent back
        index_owner->return_slice_commit(points.size(), commit_precondition);
      }
      commit_operation(true /*deactivate*/, commit_precondition);
    }

    //--------------------------------------------------------------------------
    void SliceTask::return_privileges(
        TaskContext* point_context, std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      // If we're remote, pass our privileges back to ourself
      // otherwise pass them directly back to the index owner
      if (is_remote())
        point_context->return_resources(this, context_index, preconditions);
      else if (must_epoch != nullptr)
        point_context->return_resources(
            must_epoch, context_index, preconditions);
      else
        point_context->return_resources(
            parent_ctx, context_index, preconditions);
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_point_mapped(
        PointTask* point, RtEvent child_mapped, bool shard_off)
    //--------------------------------------------------------------------------
    {
      bool done_mapping;
      {
        AutoLock o_lock(op_lock);
        // Can safely overwrite if there is already an event from a call
        // to find_intra_space_dependence
        point_mapped_events.emplace(
            std::make_pair(point->index_point, child_mapped));
        legion_assert(num_unmapped_points > 0);
        done_mapping = (--num_unmapped_points == 0);
      }
      // Send this point back to the index owner task
      if (!is_remote())
        index_owner->return_point_mapped(point->index_point, child_mapped);
      // Only need to send something back if we're not origin mapped
      else if (!is_origin_mapped())
      {
        SliceRemoteMapped rez;
        rez.serialize(index_owner);
        {
          RezCheck z(rez);
          rez.serialize(point->index_point);
          rez.serialize(child_mapped);
        }
        rez.dispatch(orig_proc.address_space());
      }
      if (done_mapping)
      {
        complete_mapping();
        // Check to see if we need to handle the remaining stage of
        // execution for this slice when origin mapped
        if (!shard_off && is_origin_mapped() && !is_remote() &&
            !is_replaying() && distribute_task())
          launch_task();
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_point_complete(ApEvent child_effects)
    //--------------------------------------------------------------------------
    {
      if (child_effects.exists())
        record_completion_effect(child_effects);
      const unsigned remaining = num_uncompleted_points.fetch_sub(1);
      legion_assert(remaining > 0);
      if (remaining == 1)
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_point_committed(RtEvent commit_precondition)
    //--------------------------------------------------------------------------
    {
      bool needs_trigger = false;
      {
        AutoLock o_lock(op_lock);
        legion_assert(num_uncommitted_points > 0);
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
    void SliceTask::handle_future_size(
        size_t future_size, const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      if (redop > 0)
        return;
      legion_assert(!elide_future_return);
      legion_assert(future_handles != nullptr);
      const std::map<DomainPoint, DistributedID>& handles =
          future_handles->handles;
      std::map<DomainPoint, DistributedID>::const_iterator finder =
          handles.find(point);
      legion_assert(finder != handles.end());
      const ContextCoordinate coordinate(future_map_coordinate, point);
      RtEvent registered;
      FutureImpl* impl = runtime->find_or_create_future(
          finder->second, parent_ctx->did, coordinate, get_provenance(),
          false /*has global reference*/, registered);
      impl->set_future_result_size(future_size, runtime->address_space);
      if (registered.exists())
      {
        AutoLock o_lock(op_lock);
        commit_preconditions.insert(registered);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::record_output_extent(
        unsigned index, const DomainPoint& color, const DomainPoint& extent)
    //--------------------------------------------------------------------------
    {
      legion_assert(index < output_regions.size());
      legion_assert(output_regions.size() == output_region_extents.size());
      legion_assert(output_regions.size() == output_region_options.size());
      legion_assert(!is_output_valid(index));
      {
        AutoLock o_lock(op_lock);
        OutputExtentMap& output_extents = output_region_extents[index];
        if (output_extents.find(color) != output_extents.end())
        {
          const OutputRequirement& req = output_regions[index];
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "A projection functor for every output requirement must be "
                << "bijective, but projection functor " << req.projection
                << " for output requirement " << index << " in " << *this
                << " mapped more than one point in the launch domain to the "
                << "same subregion of color " << color << ".";
          error.raise();
        }
        output_extents[color] = extent;
        legion_assert(output_extents.size() <= points.size());
        if (output_extents.size() < points.size())
          return;
        // Check the other output regions to see if they are done as well
        for (unsigned idx = 0; idx < output_regions.size(); idx++)
        {
          if (idx == index)
            continue;
          if (is_output_valid(idx))
            continue;
          legion_assert(output_region_extents[idx].size() <= points.size());
          if (output_region_extents[idx].size() < points.size())
            return;
        }
      }
      // If we get here then we need to send the sizes back to the index owner
      if (is_remote())
      {
        const RtUserEvent applied = Runtime::create_rt_user_event();
        // Send a message back to the owner with the output region extents
        SliceRemoteOutputExtents rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize<size_t>(output_region_extents.size());
          for (const OutputExtentMap& extents : output_region_extents)
          {
            rez.serialize<size_t>(extents.size());
            for (const std::pair<const DomainPoint, DomainPoint>& extent_pair :
                 extents)
            {
              rez.serialize(extent_pair.first);
              rez.serialize(extent_pair.second);
            }
          }
          rez.serialize(applied);
        }
        rez.dispatch(orig_proc.address_space());
        AutoLock o_lock(op_lock);
        legion_assert(num_uncompleted_points.load() > 0);
        commit_preconditions.insert(applied);
      }
      else
        index_owner->record_output_extents(output_region_extents);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceRemoteOutputExtents::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask* index_owner;
      derez.deserialize(index_owner);
      size_t num_regions;
      derez.deserialize(num_regions);
      std::vector<MultiTask::OutputExtentMap> output_region_extents(
          num_regions);
      for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
      {
        MultiTask::OutputExtentMap& extents = output_region_extents[idx1];
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
    void SliceTask::record_output_registered(
        RtEvent registered, std::set<RtEvent>& applied_events)
    //--------------------------------------------------------------------------
    {
      legion_assert(registered.exists());
      if (is_remote())
      {
        // Send a message back to the index owner about the equivalence
        // sets for the output regions being registered
        const RtUserEvent applied = Runtime::create_rt_user_event();
        SliceRemoteOutputRegistration rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize(registered);
          rez.serialize(applied);
        }
        rez.dispatch(orig_proc.address_space());
        applied_events.insert(applied);
      }
      else
        index_owner->record_output_registered(registered);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceRemoteOutputRegistration::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask* index_owner;
      derez.deserialize(index_owner);
      RtEvent registered;
      derez.deserialize(registered);
      RtUserEvent applied;
      derez.deserialize(applied);
      index_owner->record_output_registered(registered);
      Runtime::trigger_event(applied);
    }

    //--------------------------------------------------------------------------
    void SliceTask::rendezvous_concurrent_mapped(
        const DomainPoint& point, Processor target, Color color,
        RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      legion_assert(concurrent_task);
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
      legion_assert(finder != concurrent_groups.end());
      if (is_remote())
      {
        AutoLock o_lock(op_lock);
        if (precondition.exists())
          finder->second.preconditions.emplace_back(precondition);
        std::map<Processor, DomainPoint>::const_iterator proc_finder =
            finder->second.processors.find(target);
        if (proc_finder != finder->second.processors.end())
          report_concurrent_mapping_failure(target, point, proc_finder->second);
        legion_assert(concurrent_points < points.size());
        if (++concurrent_points == points.size())
          send_rendezvous_concurrent_mapped();
      }
      else
        index_owner->rendezvous_concurrent_mapped(
            point, target, color, precondition);
    }

    //--------------------------------------------------------------------------
    void SliceTask::send_rendezvous_concurrent_mapped(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_remote());
      legion_assert(concurrent_task);
      SliceConcurrentMapped rez;
      {
        RezCheck z(rez);
        rez.serialize(index_owner);
        // Count how many colors have "interesting" results
        size_t num_colors = 0;
        for (const std::pair<const Color, ConcurrentGroup>& group :
             concurrent_groups)
          if (!group.second.preconditions.empty())
            num_colors++;
        rez.serialize(num_colors);
        for (const std::pair<const Color, ConcurrentGroup>& group :
             concurrent_groups)
        {
          if (group.second.processors.empty())
            continue;
          rez.serialize(group.first);
          if (group.second.preconditions.empty())
            rez.serialize(RtEvent::NO_RT_EVENT);
          else
            rez.serialize(Runtime::merge_events(group.second.preconditions));
          rez.serialize<size_t>(group.second.processors.size());
          for (const std::pair<const Processor, DomainPoint>& proc_pair :
               group.second.processors)
          {
            rez.serialize(proc_pair.first);
            rez.serialize(proc_pair.second);
          }
        }
      }
      rez.dispatch(orig_proc.address_space());
    }

    //--------------------------------------------------------------------------
    uint64_t SliceTask::collective_lamport_allreduce(
        uint64_t lamport_clock, bool need_result)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        AutoLock o_lock(op_lock);
        if (collective_lamport_clock < lamport_clock)
          collective_lamport_clock = lamport_clock;
        legion_assert(collective_unbounded_points < points.size());
        if (!collective_lamport_clock_ready.exists() && need_result)
          collective_lamport_clock_ready = Runtime::create_rt_user_event();
        if (++collective_unbounded_points < points.size())
        {
          if (need_result)
          {
            o_lock.release();
            collective_lamport_clock_ready.wait();
          }
          return collective_lamport_clock;
        }
        // Otherwise fall through and send the message to the index owner
      }
      else
        return index_owner->collective_lamport_allreduce(
            lamport_clock, 1 /*points*/, need_result);
      SliceCollectiveRequest rez;
      {
        RezCheck z(rez);
        rez.serialize(index_owner);
        rez.serialize(collective_unbounded_points);
        rez.serialize(collective_lamport_clock);
        if (!collective_lamport_clock_ready.exists())
        {
          // Still need to make one to know when results are applied
          collective_lamport_clock_ready = Runtime::create_rt_user_event();
          rez.serialize(collective_lamport_clock_ready);
          rez.serialize<bool>(false);  // need result;
          // Put this in the commit preconditions data structure since
          // we need to capture it as part of the effects of this task
          // in case nothing ends up needing it
          AutoLock o_lock(op_lock);
          commit_preconditions.insert(collective_lamport_clock_ready);
        }
        else
        {
          rez.serialize(collective_lamport_clock_ready);
          rez.serialize<bool>(true);  // need result;
          rez.serialize(&collective_lamport_clock);
        }
      }
      rez.dispatch(orig_proc.address_space());
      if (need_result)
        collective_lamport_clock_ready.wait();
      return collective_lamport_clock;
    }

    //--------------------------------------------------------------------------
    void SliceTask::concurrent_allreduce(
        PointTask* task, ProcessorManager* manager, uint64_t lamport_clock,
        VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(task->concurrent_color);
      legion_assert(finder != concurrent_groups.end());
      {
        AutoLock o_lock(op_lock);
        if (finder->second.lamport_clock < lamport_clock)
          finder->second.lamport_clock = lamport_clock;
        if (poisoned)
          finder->second.poisoned = true;
        if (finder->second.point_tasks.empty())
          finder->second.variant = vid;
        else if (finder->second.variant != vid)
          finder->second.variant = std::min(finder->second.variant, vid);
        finder->second.point_tasks.emplace_back(std::make_pair(task, manager));
        done =
            (finder->second.point_tasks.size() == finder->second.group_points);
      }
      if (done)
      {
        if (is_remote())
        {
          SliceConcurrentRequest rez;
          {
            RezCheck z(rez);
            rez.serialize(index_owner);
            rez.serialize(this);
            rez.serialize(finder->first);
            rez.serialize(finder->second.group_points);
            rez.serialize(finder->second.lamport_clock);
            rez.serialize(finder->second.variant);
            rez.serialize(finder->second.poisoned);
          }
          rez.dispatch(orig_proc.address_space());
        }
        else
          index_owner->concurrent_allreduce(
              finder->first, this, runtime->address_space,
              finder->second.group_points, finder->second.lamport_clock,
              finder->second.variant, finder->second.poisoned);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::finish_concurrent_allreduce(
        Color color, uint64_t lamport_clock, bool poisoned, VariantID vid,
        RtBarrier concurrent_barrier)
    //--------------------------------------------------------------------------
    {
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
      legion_assert(finder != concurrent_groups.end());
      if (concurrent_barrier.exists())
        finder->second.task_barrier = concurrent_barrier;
      // Swap this vector onto the stack in case the slice task gets deleted
      // out from under us while we are finalizing things
      std::vector<std::pair<PointTask*, ProcessorManager*>> local_copy;
      local_copy.swap(finder->second.point_tasks);
      for (const std::pair<PointTask*, ProcessorManager*>& it : local_copy)
        if (must_epoch_task || it.first->check_concurrent_variant(vid))
          it.second->finalize_concurrent_task_order(
              it.first, lamport_clock, poisoned);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceConcurrentMapped::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask* owner;
      derez.deserialize(owner);
      owner->rendezvous_concurrent_mapped(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceCollectiveRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask* owner;
      derez.deserialize(owner);
      size_t total_points;
      derez.deserialize(total_points);
      uint64_t lamport_clock;
      derez.deserialize(lamport_clock);
      RtUserEvent done;
      derez.deserialize(done);
      bool need_result;
      derez.deserialize<bool>(need_result);

      const uint64_t result = owner->collective_lamport_allreduce(
          lamport_clock, total_points, need_result);
      if (need_result)
      {
        uint64_t* target;
        derez.deserialize(target);
        SliceCollectiveResponse rez;
        {
          RezCheck z2(rez);
          rez.serialize(target);
          rez.serialize(result);
          rez.serialize(done);
        }
        rez.dispatch(source);
      }
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceCollectiveResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      uint64_t* target;
      derez.deserialize(target);
      derez.deserialize(*target);
      RtUserEvent done;
      derez.deserialize(done);
      Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceConcurrentRequest::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask* owner;
      derez.deserialize(owner);
      SliceTask* slice;
      derez.deserialize(slice);
      Color color;
      derez.deserialize(color);
      size_t total_points;
      derez.deserialize(total_points);
      uint64_t lamport_clock;
      derez.deserialize(lamport_clock);
      VariantID variant;
      derez.deserialize(variant);
      bool poisoned;
      derez.deserialize<bool>(poisoned);
      owner->concurrent_allreduce(
          color, slice, source, total_points, lamport_clock, variant, poisoned);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceConcurrentResponse::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      SliceTask* slice;
      derez.deserialize(slice);
      Color color;
      derez.deserialize(color);
      RtBarrier barrier;
      derez.deserialize(barrier);
      uint64_t lamport_clock;
      derez.deserialize(lamport_clock);
      VariantID vid;
      derez.deserialize(vid);
      bool poisoned;
      derez.deserialize<bool>(poisoned);
      slice->finish_concurrent_allreduce(
          color, lamport_clock, poisoned, vid, barrier);
    }

    //--------------------------------------------------------------------------
    void SliceTask::forward_completion_effects(void)
    //--------------------------------------------------------------------------
    {
      for (PointTask* const pt : points)
        pt->forward_completion_effects(index_owner);
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_complete(Serializer& rez, ApEvent slice_effects)
    //--------------------------------------------------------------------------
    {
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize<size_t>(points.size());
      rez.serialize(slice_effects);
      // Now pack up the future results
      if (redop > 0)
      {
        if (deterministic_redop)
        {
          legion_assert(reduction_instance == nullptr);
          // Might have no temporary futures if this task was predicated
          // and the predicate resolved to false
          legion_assert(
              (temporary_futures.size() == points.size()) ||
              temporary_futures.empty());
          legion_assert(reduction_fold_effects.empty());
          rez.serialize<size_t>(temporary_futures.size());
          for (const std::pair<
                   const DomainPoint, std::pair<FutureInstance*, ApEvent>>&
                   future_pair : temporary_futures)
          {
            rez.serialize(future_pair.first);
            if (!future_pair.second.first->pack_instance(
                    rez, future_pair.second.second, true /*pack ownership*/))
              rez.serialize(future_pair.second.second);
          }
        }
        else
        {
          if (serdez_redop_fns != nullptr)
          {
            legion_assert(reduction_instance == nullptr);
            legion_assert(reduction_fold_effects.empty());
            // Easy case just for serdez, we just pack up the local buffer
            rez.serialize(serdez_redop_state_size);
            if (serdez_redop_state_size > 0)
              rez.serialize(serdez_redop_state, serdez_redop_state_size);
          }
          else
          {
            // We might not have a reduction instance if this task was
            // predicated and ended up predicating false
            legion_assert(
                (reduction_instance != nullptr) || false_guard.exists());
            legion_assert(
                (reduction_instance != nullptr) ==
                (reduction_instance_point.get_dim() > 0));
            rez.serialize(reduction_instance_point);
            if (!reduction_fold_effects.empty())
              // All the reduction fold effects dominate the
              // reduction_instance_precondition so we can just
              // overwrite it without including it in the merger
              reduction_instance_precondition =
                  Runtime::merge_events(nullptr, reduction_fold_effects);
            if ((reduction_instance != nullptr) &&
                !reduction_instance.load()->pack_instance(
                    rez, reduction_instance_precondition,
                    true /*pack ownership*/))
              rez.serialize(reduction_instance_precondition);
          }
        }
        if (reduction_metadata != nullptr)
        {
          rez.serialize(reduction_metasize);
          rez.serialize(reduction_metadata, reduction_metasize);
        }
        else
          rez.serialize<size_t>(0);
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::pack_remote_commit(
        Serializer& rez, RtEvent applied_condition)
    //--------------------------------------------------------------------------
    {
      rez.serialize(index_owner);
      RezCheck z(rez);
      rez.serialize(points.size());
      rez.serialize(applied_condition);
      // Serialize the privilege state
      pack_resources_return(rez, context_index);
    }

    //--------------------------------------------------------------------------
    void SliceTask::receive_resources(
        uint64_t return_index, std::map<LogicalRegion, unsigned>& created_regs,
        std::vector<DeletedRegion>& deleted_regs,
        std::set<std::pair<FieldSpace, FieldID>>& created_fids,
        std::vector<DeletedField>& deleted_fids,
        std::map<FieldSpace, unsigned>& created_fs,
        std::map<FieldSpace, std::set<LogicalRegion>>& latent_fs,
        std::vector<DeletedFieldSpace>& deleted_fs,
        std::map<IndexSpace, unsigned>& created_is,
        std::vector<DeletedIndexSpace>& deleted_is,
        std::map<IndexPartition, unsigned>& created_partitions,
        std::vector<DeletedPartition>& deleted_partitions,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      merge_received_resources(
          created_regs, deleted_regs, created_fids, deleted_fids, created_fs,
          latent_fs, deleted_fs, created_is, deleted_is, created_partitions,
          deleted_partitions);
    }

    //--------------------------------------------------------------------------
    void SliceTask::expand_replay_slices(std::list<SliceTask*>& slices)
    //--------------------------------------------------------------------------
    {
      legion_assert(!points.empty());
      legion_assert(is_origin_mapped());
      // For each point give it its own slice owner in case we need to
      // to move it remotely as part of the replay
      while (points.size() > 1)
      {
        PointTask* point = points.back();
        points.pop_back();
        SliceTask* new_owner = clone_as_slice_task(
            internal_space, current_proc, false /*recurse*/,
            false /*stealable*/);
        point->slice_owner = new_owner;
        new_owner->points.emplace_back(point);
        new_owner->num_unmapped_points = 1;
        new_owner->num_uncompleted_points.store(1);
        new_owner->num_uncommitted_points = 1;
        if (concurrent_task)
        {
          std::map<Color, ConcurrentGroup>::iterator finder =
              new_owner->concurrent_groups.find(point->concurrent_color);
          legion_assert(finder != new_owner->concurrent_groups.end());
          finder->second.group_points = 1;
        }
        slices.emplace_back(new_owner);
      }
      // Always add ourselves as the last point
      slices.emplace_back(this);
      num_unmapped_points = points.size();
      num_uncompleted_points.store(points.size());
      num_uncommitted_points = points.size();
      if (concurrent_task)
      {
        std::map<Color, ConcurrentGroup>::iterator finder =
            concurrent_groups.find(points.back()->concurrent_color);
        legion_assert(finder != concurrent_groups.end());
        finder->second.group_points = 1;
      }
    }

    //--------------------------------------------------------------------------
    void SliceTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      for (PointTask* const pt : points) pt->trigger_replay();
    }

    //--------------------------------------------------------------------------
    void SliceTask::complete_replay(ApEvent instance_ready_event)
    //--------------------------------------------------------------------------
    {
      for (PointTask* const pt : points)
        pt->complete_replay(instance_ready_event);
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::find_intra_space_dependence(const DomainPoint& point)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
      {
        AutoLock o_lock(op_lock);
        std::map<DomainPoint, RtEvent>::const_iterator finder =
            point_mapped_events.find(point);
        // If we've already got it then we're done
        if (finder != point_mapped_events.end())
          return finder->second;
        legion_assert(!points.empty());
#if 0
        // This optimization is no longer safe because we don't know if
        // some points are going to be sent away remotely later
        // Next see if it is one of our local points
        for (PointTask* const pt : points)
        {
          if (pt->index_point != point)
            continue;
          // Don't save this in our intra_space_dependences data structure!
          // Doing so could mess up our optimization for detecting when 
          // we need to send dependences back to the origin
          // See SliceTask::record_intra_space_dependence
          return pt->get_mapped_event();
        }
#endif
        const RtUserEvent temp_event = Runtime::create_rt_user_event();
        // Send the message to the owner to go find it
        SliceFindIntraDependence rez;
        {
          RezCheck z(rez);
          rez.serialize(index_owner);
          rez.serialize(point);
          rez.serialize(temp_event);
        }
        rez.dispatch(orig_proc.address_space());
        // Save this is for ourselves
        point_mapped_events[point] = temp_event;
        return temp_event;
      }
      else
        return index_owner->find_intra_space_dependence(point);
    }

    //--------------------------------------------------------------------------
    size_t SliceTask::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(is_remote());
      return points.size();
    }

    //--------------------------------------------------------------------------
    bool SliceTask::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(!is_remote());
      return index_owner->find_shard_participants(shards);
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        return MultiTask::rendezvous_collective_versioning_analysis(
            index, handle, tracker, runtime->address_space, mask,
            parent_req_index);
      else
        return index_owner->rendezvous_collective_versioning_analysis(
            index, handle, tracker, runtime->address_space, mask,
            parent_req_index);
    }

    //--------------------------------------------------------------------------
    void SliceTask::perform_replicate_collective_versioning(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, RegionVersioning>& to_perform)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        MultiTask::rendezvous_collective_versioning_analysis(
            index, parent_req_index, to_perform);
      else
        index_owner->rendezvous_collective_versioning_analysis(
            index, parent_req_index, to_perform);
    }

    //--------------------------------------------------------------------------
    void SliceTask::convert_replicate_collective_views(
        const RendezvousKey& key,
        std::map<LogicalRegion, CollectiveRendezvous>& rendezvous)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        MultiTask::rendezvous_collective_mapping(key, rendezvous);
      else
        index_owner->rendezvous_collective_mapping(key, rendezvous);
    }

    //--------------------------------------------------------------------------
    void SliceTask::finalize_collective_versioning_analysis(
        unsigned index, unsigned parent_req_index,
        op::map<LogicalRegion, RegionVersioning>& to_perform)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_remote());
      SliceRemoteVersioningCollective rez;
      {
        RezCheck z(rez);
        rez.serialize(index_owner);
        rez.serialize(index);
        rez.serialize<size_t>(points.size());
        rez.serialize<size_t>(to_perform.size());
        for (const std::pair<const LogicalRegion, RegionVersioning>&
                 version_pair : to_perform)
        {
          rez.serialize(version_pair.first);
          legion_assert(version_pair.second.ready_event.exists());
          rez.serialize(version_pair.second.ready_event);
          rez.serialize<size_t>(version_pair.second.trackers.size());
          for (const std::pair<
                   const std::pair<AddressSpaceID, EqSetTracker*>, FieldMask>&
                   tracker_pair : version_pair.second.trackers)
          {
            rez.serialize(tracker_pair.first.first);
            rez.serialize(tracker_pair.first.second);
            rez.serialize(tracker_pair.second);
          }
        }
        if (to_perform.empty())
        {
          // If we don't have any local points depending on the result
          // then we need to pack an event to make sure this message gets
          // there before the index task is cleaned up
          const RtUserEvent done_event = Runtime::create_rt_user_event();
          rez.serialize(done_event);
          AutoLock o_lock(op_lock);
          commit_preconditions.insert(done_event);
        }
      }
      rez.dispatch(orig_proc.address_space());
    }

    //--------------------------------------------------------------------------
    RtEvent SliceTask::convert_collective_views(
        unsigned requirement_index, unsigned analysis_index,
        LogicalRegion region, const InstanceSet& targets,
        InnerContext* physical_ctx, CollectiveMapping*& analysis_mapping,
        bool& first_local,
        op::vector<op::FieldMaskMap<InstanceView>>& target_views,
        std::map<InstanceView*, size_t>& collective_arrivals)
    //--------------------------------------------------------------------------
    {
      if (is_remote())
        return MultiTask::convert_collective_views(
            requirement_index, analysis_index, region, targets, physical_ctx,
            analysis_mapping, first_local, target_views, collective_arrivals);
      else
        return index_owner->convert_collective_views(
            requirement_index, analysis_index, region, targets, physical_ctx,
            analysis_mapping, first_local, target_views, collective_arrivals);
    }

    //--------------------------------------------------------------------------
    void SliceTask::rendezvous_collective_mapping(
        unsigned requirement_index, unsigned analysis_index,
        LogicalRegion region, RendezvousResult* result, AddressSpaceID source,
        const op::vector<std::pair<DistributedID, FieldMask>>& insts)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_remote());
      legion_assert(source == runtime->address_space);
      // Send this back to the owner node
      SliceRemoteCollective rez;
      {
        RezCheck z(rez);
        rez.serialize(index_owner);
        rez.serialize(requirement_index);
        rez.serialize(analysis_index);
        rez.serialize(region);
        rez.serialize(result);
        rez.serialize<size_t>(insts.size());
        for (const std::pair<DistributedID, FieldMask>& inst : insts)
        {
          rez.serialize(inst.first);
          rez.serialize(inst.second);
        }
      }
      rez.dispatch(orig_proc.address_space());
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceRemoteCollective::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask* index_owner;
      derez.deserialize(index_owner);
      unsigned requirement_index, analysis_index;
      derez.deserialize(requirement_index);
      derez.deserialize(analysis_index);
      LogicalRegion region;
      derez.deserialize(region);
      SliceTask::RendezvousResult* result;
      derez.deserialize(result);
      size_t num_insts;
      derez.deserialize(num_insts);
      op::vector<std::pair<DistributedID, FieldMask>> instances(num_insts);
      for (std::pair<DistributedID, FieldMask>& instance : instances)
      {
        derez.deserialize(instance.first);
        derez.deserialize(instance.second);
      }

      index_owner->rendezvous_collective_mapping(
          requirement_index, analysis_index, region, result, source, instances);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceRemoteVersioningCollective::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask* index_owner;
      derez.deserialize(index_owner);
      unsigned index;
      derez.deserialize(index);
      size_t total_points;
      derez.deserialize(total_points);
      index_owner->unpack_slice_collective_versioning_rendezvous(
          derez, index, total_points);
    }

  }  // namespace Internal
}  // namespace Legion
