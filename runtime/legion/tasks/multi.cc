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

#include "legion/tasks/multi.h"
#include "legion/contexts/inner.h"
#include "legion/api/future_impl.h"
#include "legion/nodes/index.h"
#include "legion/managers/mapper.h"
#include "legion/tasks/index.h"
#include "legion/tasks/slice.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Multi Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MultiTask::MultiTask(void)
      : PointwiseAnalyzable<CollectiveViewCreator<TaskOp> >()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MultiTask::~MultiTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void MultiTask::activate(void)
    //--------------------------------------------------------------------------
    {
      PointwiseAnalyzable<CollectiveViewCreator<TaskOp> >::activate();
      launch_space = nullptr;
      future_map_coordinate = 0;
      future_handles = nullptr;
      internal_space = IndexSpace::NO_SPACE;
      sliced = false;
      redop = 0;
      deterministic_redop = false;
      reduction_op = nullptr;
      serdez_redop_fns = nullptr;
      serdez_redop_state = nullptr;
      serdez_redop_state_size = 0;
      reduction_metadata = nullptr;
      reduction_metasize = 0;
      reduction_instance = nullptr;
      concurrent_functor = 0;
      concurrent_points = 0;
      collective_lamport_clock = 0;
      collective_unbounded_points = 0;
      collective_lamport_clock_ready = RtUserEvent::NO_RT_USER_EVENT;
      children_commit_invoked = false;
    }

    //--------------------------------------------------------------------------
    void MultiTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      if (implicit_profiler != nullptr)
        implicit_profiler->register_multi_task(this, task_id);
      if (remove_launch_space_reference(launch_space))
        delete launch_space;
      if ((future_handles != nullptr) && future_handles->remove_reference())
        delete future_handles;
      redop_initial_value = Future();
      // Remove our reference to the future map
      future_map = FutureMap();
      if (reduction_instance != nullptr)
        delete reduction_instance.load();
      reduction_fold_effects.clear();
      if (serdez_redop_state != nullptr)
        free(serdez_redop_state);
      if (reduction_metadata != nullptr)
        free(reduction_metadata);
      if (!temporary_futures.empty())
      {
        for (const std::pair<
                 const DomainPoint, std::pair<FutureInstance*, ApEvent> >&
                 temp_future : temporary_futures)
          delete temp_future.second.first;
        temporary_futures.clear();
      }
      concurrent_groups.clear();
      // Remove our reference to the point arguments
      point_arguments = FutureMap();
      point_futures.clear();
      output_region_options.clear();
      slices.clear();
      predicate_false_result.clear();
      predicate_false_future = Future();
      point_mapped_events.clear();
      PointwiseAnalyzable<CollectiveViewCreator<TaskOp> >::deactivate(freeop);
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
      legion_assert(!sliced);
      sliced = true;
      stealable = false;  // cannot steal something that has been sliced
      Mapper::SliceTaskInput input;
      Mapper::SliceTaskOutput output;
      input.domain_is = internal_space;
      if (sharding_space.exists())
        input.sharding_is = sharding_space;
      else
        input.sharding_is = launch_space->handle;
      runtime->find_domain(internal_space, input.domain);
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_slice_task(this, input, output);
      if (output.slices.empty())
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'slice_task' "
              << "call on mapper " << mapper->get_mapper_name()
              << ". Mapper failed to specify any slices for task "
              << get_task_name() << " (ID " << get_unique_id() << ").";
        error.raise();
      }
      size_t total_points = 0;
      for (Mapper::TaskSlice& slice : output.slices)
      {
        if (!slice.proc.exists())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'slice_task' "
                << "on mapper " << mapper->get_mapper_name()
                << ". Mapper returned a slice for task " << get_task_name()
                << " (ID " << get_unique_id() << ") with an invalid processor "
                << slice.proc.id << ".";
          error.raise();
        }
        // Check to see if we need to get an index space for this domain
        if (!slice.domain_is.exists())
        {
          // Skip any empty slices
          if (slice.domain.get_volume() == 0)
            continue;
          if (slice.domain.dense())
            slice.domain.is_type = internal_space.get_type_tag();
          else if (slice.domain.is_type != internal_space.get_type_tag())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of 'slice_task' "
                  << "on mapper " << mapper->get_mapper_name()
                  << ". Mapper returned sparse slice domain for task "
                  << get_task_name() << " (UID " << get_unique_id()
                  << ") with a sparsity map of a different type (type_tag="
                  << slice.domain.is_type << ") than original index space "
                  << "to be sliced (type_tag=" << internal_space.get_type_tag()
                  << ").";
            error.raise();
          }
          slice.domain_is = runtime->find_or_create_index_slice_space(
              slice.domain, slice.take_ownership, get_provenance());
        }
        else if (
            slice.domain_is.get_type_tag() != internal_space.get_type_tag())
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output from invocation of 'slice_task' "
                << "on mapper " << mapper->get_mapper_name()
                << ". Mapper returned slice index space "
                << slice.domain_is.get_id() << " for task " << get_task_name()
                << " (UID " << get_unique_id()
                << ") with a different type than original index space to be "
                   "sliced.";
          error.raise();
        }
        if (runtime->safe_mapper)
        {
          // Check to make sure the domain is not empty
          Domain& d = slice.domain;
          if ((d == Domain::NO_DOMAIN) && slice.domain_is.exists())
            runtime->find_domain(slice.domain_is, d);
          bool empty = false;
          size_t volume = d.get_volume();
          if (volume == 0)
            empty = true;
          else
            total_points += volume;
          if (empty)
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Invalid mapper output from invocation of 'slice_task' "
                  << "on mapper " << mapper->get_mapper_name()
                  << ". Mapper returned an empty slice for task "
                  << get_task_name() << " (ID " << get_unique_id() << ").";
            error.raise();
          }
        }
        SliceTask* new_slice = this->clone_as_slice_task(
            slice.domain_is, slice.proc, slice.recurse, slice.stealable);
        slices.emplace_back(new_slice);
      }
      // If the volumes don't match, then something bad happend in the mapper
      if (runtime->safe_mapper && (total_points != input.domain.get_volume()))
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'slice_task' "
              << "on mapper " << mapper->get_mapper_name()
              << ". Mapper returned slices with a total volume "
              << long(total_points)
              << " that does not match the expected volume of "
              << input.domain.get_volume() << " when slicing task "
              << get_task_name() << " (ID " << get_unique_id() << ").";
        error.raise();
      }
      if (output.verify_correctness || runtime->safe_mapper)
      {
        std::vector<IndexSpace> slice_spaces;
        slice_spaces.reserve(slices.size());
        for (unsigned idx = 0; idx < output.slices.size(); idx++)
          if (output.slices[idx].domain_is.exists())
            slice_spaces.push_back(output.slices[idx].domain_is);
        IndexSpaceNode* node = runtime->get_node(internal_space);
        node->validate_slicing(slice_spaces, this, mapper);
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
      std::vector<RtEvent> wait_for;
      std::list<SliceTask*>::const_iterator it = slices.begin();
      while (true)
      {
        SliceTask* slice = *it;
        // Have to update this before launching the task to avoid
        // the clean-up race
        it++;
        const bool done = (it == slices.end());
        // Dumb case for must epoch operations, we need these to
        // be mapped immediately, mapper be damned
        if (must_epoch != nullptr)
        {
          TriggerTaskArgs trigger_args(slice, parent_ctx->did);
          RtEvent done = runtime->issue_runtime_meta_task(
              trigger_args, LG_THROUGHPUT_WORK_PRIORITY);
          wait_for.emplace_back(done);
        }
        // If we're replaying this for for a trace then don't even
        // bother asking the mapper about when to map this
        else if (is_replaying())
          slice->trigger_mapping();
        // Figure out whether this task is local or remote
        else if (!runtime->is_local(slice->target_proc))
        {
          // We can only send it away if it is not origin mapped
          // otherwise it has to stay here until it is fully mapped
          if (!slice->is_origin_mapped())
            runtime->send_task(slice);
          else
            slice->trigger_mapping();
        }
        else
        {
          slice->set_current_proc(slice->target_proc);
          slice->trigger_mapping();
        }
        if (done)
          break;
      }
      // Must-epoch operations are nasty little beasts and have
      // to wait for the effects to finish before returning
      if (!wait_for.empty())
        Runtime::merge_events(wait_for).wait();
    }

    //--------------------------------------------------------------------------
    void MultiTask::clone_multi_from(
        MultiTask* rhs, IndexSpace is, Processor p, bool recurse,
        bool stealable)
    //--------------------------------------------------------------------------
    {
      legion_assert(this->launch_space == nullptr);
      legion_assert(this->future_handles == nullptr);
      this->clone_task_op_from(rhs, p, stealable);
      this->index_domain = rhs->index_domain;
      this->launch_space = rhs->launch_space;
      add_launch_space_reference(this->launch_space);
      this->future_map_coordinate = rhs->future_map_coordinate;
      this->future_handles = rhs->future_handles;
      if (this->future_handles != nullptr)
        this->future_handles->add_reference();
      this->internal_space = is;
      this->future_map = rhs->future_map;
      this->must_epoch_task = rhs->must_epoch_task;
      this->sliced = !recurse;
      this->pointwise_dependences = rhs->pointwise_dependences;
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
      if (!elide_future_return)
      {
        this->predicate_false_future = rhs->predicate_false_future;
        if (rhs->predicate_false_result.get_size() > 0)
          this->predicate_false_result.save_buffer(
              rhs->predicate_false_result.get_buffer(),
              rhs->predicate_false_result.get_size());
      }
      if (rhs->concurrent_task)
      {
        this->concurrent_functor = rhs->concurrent_functor;
        this->concurrent_groups = rhs->concurrent_groups;
      }
    }

    //--------------------------------------------------------------------------
    RtBarrier MultiTask::get_concurrent_task_barrier(Color color) const
    //--------------------------------------------------------------------------
    {
      std::map<Color, ConcurrentGroup>::const_iterator finder =
          concurrent_groups.find(color);
      legion_assert(finder != concurrent_groups.end());
      return finder->second.task_barrier;
    }

    //--------------------------------------------------------------------------
    Domain MultiTask::get_slice_domain(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(internal_space.exists());
      Domain result;
      runtime->find_domain(internal_space, result);
      return result;
    }

    //--------------------------------------------------------------------------
    void MultiTask::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
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
              perform_mapping();
          }
          else
            slice_index_space();
        }
      }
      else
      {
        // Not remote
        if (must_epoch == nullptr)
          premap_task();
        if (is_origin_mapped())
        {
          if (is_sliced())
          {
            if (must_epoch == nullptr)
              perform_mapping();
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
              perform_mapping();
            else
              slice_index_space();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    bool MultiTask::is_pointwise_analyzable(void) const
    //--------------------------------------------------------------------------
    {
      if (concurrent_task)
        return false;
      if (!check_collective_regions.empty())
        return false;
      return PointwiseAnalyzable<
          CollectiveViewCreator<TaskOp> >::is_pointwise_analyzable();
    }

    //--------------------------------------------------------------------------
    void MultiTask::pack_multi_task(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_base_task(rez, target);
      rez.serialize(launch_space->handle);
      rez.serialize(sliced);
      rez.serialize(redop);
      if (redop > 0)
        rez.serialize<bool>(deterministic_redop);
      else if (future_handles != nullptr)
      {
        // Only pack the IDs for our local points
        IndexSpaceNode* node = runtime->get_node(internal_space);
        Domain local_domain = node->get_tight_domain();
        size_t local_size = local_domain.get_volume();
        rez.serialize(local_size);
        const std::map<DomainPoint, DistributedID>& handles =
            future_handles->handles;
        legion_assert(local_size <= handles.size());
        if (local_size < handles.size())
        {
          for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
          {
            std::map<DomainPoint, DistributedID>::const_iterator finder =
                handles.find(itr.p);
            legion_assert(finder != handles.end());
            rez.serialize(finder->first);
            rez.serialize(finder->second);
          }
        }
        else
        {
          for (const std::pair<const DomainPoint, DistributedID>& handle_pair :
               handles)
          {
            rez.serialize(handle_pair.first);
            rez.serialize(handle_pair.second);
          }
        }
        rez.serialize(future_map_coordinate);
      }
      else
        rez.serialize<size_t>(0);
      if (!output_region_options.empty())
      {
        rez.serialize<size_t>(output_region_options.size());
        for (const OutputOptions& option : output_region_options)
          rez.serialize(option);
      }
      else
        rez.serialize<size_t>(0);
      if (concurrent_task)
      {
        rez.serialize(concurrent_functor);
        rez.serialize<size_t>(concurrent_groups.size());
        for (const std::pair<const Color, ConcurrentGroup>& group_pair :
             concurrent_groups)
        {
          rez.serialize(group_pair.first);
          if (is_replaying())
            rez.serialize(group_pair.second.precondition.traced);
          else
            rez.serialize(group_pair.second.precondition.interpreted);
        }
      }
      if (!is_origin_mapped())
      {
        rez.serialize<size_t>(pointwise_dependences.size());
        for (const std::pair<const unsigned, std::vector<PointwiseDependence> >&
                 pit : pointwise_dependences)
        {
          rez.serialize(pit.first);
          rez.serialize<size_t>(pit.second.size());
          for (const PointwiseDependence& dependence : pit.second)
            dependence.serialize(rez);
        }
      }
    }

    //--------------------------------------------------------------------------
    void MultiTask::unpack_multi_task(
        Deserializer& derez, std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      unpack_base_task(derez, ready_events);
      IndexSpace launch_handle;
      derez.deserialize(launch_handle);
      legion_assert(launch_space == nullptr);
      launch_space = runtime->get_node(launch_handle);
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
        legion_assert(future_handles == nullptr);
        size_t num_handles;
        derez.deserialize(num_handles);
        if (num_handles > 0)
        {
          future_handles = new FutureHandles;
          future_handles->add_reference();
          std::map<DomainPoint, DistributedID>& handles =
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
      }
      if (concurrent_task)
      {
        derez.deserialize(concurrent_functor);
        size_t num_colors;
        derez.deserialize(num_colors);
        for (unsigned idx = 0; idx < num_colors; idx++)
        {
          Color color;
          derez.deserialize(color);
          if (is_replaying())
            derez.deserialize(concurrent_groups[color].precondition.traced);
          else
            derez.deserialize(
                concurrent_groups[color].precondition.interpreted);
        }
      }
      if (!is_origin_mapped())
      {
        size_t num_pointwise;
        derez.deserialize(num_pointwise);
        for (unsigned idx1 = 0; idx1 < num_pointwise; idx1++)
        {
          unsigned index;
          derez.deserialize(index);
          std::vector<PointwiseDependence>& dependences =
              pointwise_dependences[index];
          size_t num_dependences;
          derez.deserialize(num_dependences);
          dependences.resize(num_dependences);
          for (unsigned idx2 = 0; idx2 < num_dependences; idx2++)
            dependences[idx2].deserialize(derez);
        }
      }
    }

    //--------------------------------------------------------------------------
    bool MultiTask::fold_reduction_future(
        FutureInstance* instance, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // Apply the reduction operation
      legion_assert(reduction_op != nullptr);
      // Perform the reduction, see if we have to do serdez reductions
      if (serdez_redop_fns != nullptr)
      {
        // If this instance is not meta-visible we need to copy
        // it to a local buffer here
        FutureInstance* bounce_instance = nullptr;
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
          void* bounce_buffer = malloc(instance->size);
          bounce_instance = FutureInstance::create_local(
              bounce_buffer, instance->size, true /*own*/);
#ifdef __GNUC__
#if __GNUC__ >= 11
#pragma GCC diagnostic pop
#endif
#endif
          // Wait for the data here to be ready
          const ApEvent ready =
              bounce_instance->copy_from(instance, this, effects);
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
          if (serdez_redop_state == nullptr)
            serdez_redop_fns->init_fn(
                reduction_op, serdez_redop_state, serdez_redop_state_size);
          serdez_redop_fns->fold_fn(
              reduction_op, serdez_redop_state, serdez_redop_state_size,
              instance->get_data());
        }
        if (bounce_instance != nullptr)
          delete bounce_instance;
        return true;
      }
      else
      {
        legion_assert(reduction_instance != nullptr);
        if (effects.exists())
        {
          if (reduction_instance_precondition.exists())
            effects = Runtime::merge_events(
                nullptr, effects, reduction_instance_precondition);
        }
        else
          effects = reduction_instance_precondition;
        if (!deterministic_redop)
        {
          AutoLock o_lock(op_lock);
          const ApEvent done = reduction_instance.load()->reduce_from(
              instance, this, redop, reduction_op, false /*exclusive*/,
              effects);
          if (done.exists())
          {
            reduction_fold_effects.emplace_back(done);
            return false;
          }
          else
            return true;
        }
        else
        {
          // No need for the lock since we know the caller is ensuring order
          reduction_instance_precondition =
              reduction_instance.load()->reduce_from(
                  instance, this, redop, reduction_op, true /*exclusive*/,
                  effects);
          return !reduction_instance_precondition.exists();
        }
      }
    }

    //--------------------------------------------------------------------------
    void MultiTask::report_concurrent_mapping_failure(
        Processor proc, const DomainPoint& one, const DomainPoint& two) const
    //--------------------------------------------------------------------------
    {
      MapperManager* bad_mapper = mapper;
      if (bad_mapper == nullptr)
        bad_mapper = runtime->find_mapper(current_proc, map_id);
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "Mapper " << bad_mapper->get_mapper_name()
            << " performed illegal mapping of concurrent index space task "
            << *this << " by mapping points " << one << " and " << two
            << " to the same " << proc.kind() << " processor " << proc.id
            << ". All point tasks must be mapped to different processors for "
            << "concurrent execution of index space tasks.";
      error.raise();
    }

  }  // namespace Internal
}  // namespace Legion
