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

#include "legion/tasks/index.h"
#include "legion/contexts/replicate.h"
#include "legion/api/argument_map_impl.h"
#include "legion/api/functors_impl.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/nodes/index.h"
#include "legion/operations/mustepoch.h"
#include "legion/tasks/individual.h"
#include "legion/tasks/slice.h"
#include "legion/tracing/recognizer.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Index Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    IndexTask::IndexTask(void) : MultiTask()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    IndexTask::~IndexTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void IndexTask::activate(void)
    //--------------------------------------------------------------------------
    {
      MultiTask::activate();
      serdez_redop_fns = nullptr;
      total_points = 0;
      mapped_points = 0;
      completed_points = 0;
      committed_points = 0;
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
      reduction_instance = nullptr;  // we don't own this so clear it
      for (std::pair<const Color, ConcurrentGroup>& group : concurrent_groups)
        if (group.second.task_barrier.exists())
          group.second.task_barrier.destroy_barrier();
      if (!origin_mapped_slices.empty())
      {
        for (SliceTask* const slice : origin_mapped_slices) slice->deactivate();
        origin_mapped_slices.clear();
      }
      if (!reduction_instances.empty())
      {
        for (FutureInstance* const instance : reduction_instances)
          delete instance;
        reduction_instances.clear();
      }
      serdez_redop_targets.clear();
      // Remove our reference to the reduction future
      reduction_future = Future();
      reduction_future_size.reset();
      map_applied_conditions.clear();
      output_preconditions.clear();
      commit_preconditions.clear();
      version_infos.clear();
      interfering_requirements.clear();
      point_requirements.clear();
      output_region_states.clear();
      output_pointwise_dependences.clear();
      legion_assert(pending_pointwise_dependences.empty());
      MultiTask::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_output_extent(
        unsigned index, const DomainPoint& color, const DomainPoint& extent)
    //--------------------------------------------------------------------------
    {
      legion_assert(index < output_regions.size());
      legion_assert(index < output_region_states.size());
      legion_assert(output_regions.size() == output_region_options.size());
      legion_assert(!is_output_bounded(index));
      const OutputOptions& options = output_region_options[index];
      legion_assert(!options.bounded_requirement());
      OutputRegionState& state = output_region_states[index];
      // Should always be able to set the domain for the color when we're done
      bool done;
      std::vector<Domain> done_colors;
      {
        AutoLock o_lock(op_lock);
        if (!state.points.emplace(std::make_pair(color, extent)).second)
        {
          const OutputRequirement& req = output_regions[index];
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "A projection functor for every output requirement must "
                << "be bijective, but projection functor " << req.projection
                << " for output requirement " << index << " in task " << *this
                << " mapped more than one point in the launch domain to the "
                << "same subregion of color " << color << ".";
          error.raise();
        }
        if (options.global_indexing())
        {
          legion_assert(color.get_dim() == extent.get_dim());
          // Iterate over each of the dimensions
          for (int dim = 0; dim < color.get_dim(); dim++)
          {
            const size_t color_index =
                color[dim] - state.global_color_space.lo()[dim];
            std::vector<coord_t>& dim_extents = state.extents[dim];
            legion_assert(color_index < dim_extents.size());
            if (0 <= dim_extents[color_index])
            {
              // Already set, so there's nothing to do here other than
              // to check that the extent matches what we expect
              if (dim_extents[color_index] != extent[dim])
              {
                Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
                error << "Point task " << color
                      << " returned an output of extent " << extent[dim]
                      << " for dimension " << dim
                      << ", but an adjacent point task returned an output "
                      << "of extent " << dim_extents[dim] << ". Please make "
                      << "sure the outputs from point tasks are aligned.";
                error.raise();
              }
              continue;
            }
            else  // We're a new extent, so we can save it
              dim_extents[color_index] = extent[dim];
            // Check to see if the offset is ready
            std::vector<coord_t>& dim_offsets = state.offsets[dim];
            legion_assert(color_index < dim_offsets.size());
            // If the previous offset is not ready yet there's nothing to do
            if (dim_offsets[color_index] < 0)
              continue;
            // Ripple carry add the extents forward as many times as we can
            // and compute the new rectangle of points we need to update
            // using the new extent(s)
            DomainPoint lo, hi;
            lo = state.global_color_space.lo();
            lo[dim] += color_index;
            hi = lo;
            for (unsigned idx = color_index; idx < dim_extents.size(); idx++)
            {
              if (dim_extents[idx] < 0)
                break;
              dim_offsets[idx + 1] = dim_offsets[idx] + dim_extents[idx];
              hi[dim]++;
            }
            hi[dim]--;  // Make sure it is inclusive
            // For all the other dimensions fill in their sizes based on
            // the extents that have been previously set
            // Handle the case where one dimension doesn't have any extents yet
            bool empty_dim = false;
            for (int dim2 = 0; dim2 < color.get_dim(); dim2++)
            {
              if (dim == dim2)
                continue;
              // Count the dim_size we have here
              unsigned dim_size = 0;
              const std::vector<coord_t>& other_extents = state.extents[dim2];
              for (unsigned idx = 0; idx < other_extents.size(); idx++)
              {
                if (other_extents[idx] < 0)
                  break;
                dim_size++;
              }
              if (dim_size == 0)
              {
                empty_dim = true;
                break;
              }
              hi[dim2] = lo[dim2] + dim_size - 1;  // Make sure it's inclusive
            }
            if (!empty_dim)
              done_colors.emplace_back(Domain(lo, hi));
          }
        }
        done = (state.points.size() == total_points);
      }
      // First iterate over any done points and assign their local results
      if (!done_colors.empty())
      {
        IndexPartNode* part = runtime->get_node(
            output_regions[index].partition.get_index_partition());
        for (std::vector<Domain>::const_iterator it = done_colors.begin();
             it != done_colors.end(); it++)
        {
          for (Domain::DomainPointIterator itr(*it); itr; itr++)
          {
            DomainPoint lo, hi;
            lo.dim = color.dim;
            hi.dim = color.dim;
            for (int dim = 0; dim < color.dim; dim++)
            {
              const unsigned color_index =
                  (*itr)[dim] - state.global_color_space.lo()[dim];
              lo[dim] = state.offsets[dim][color_index];
              hi[dim] = state.offsets[dim][color_index + 1] - 1;  // inclusive
            }
            IndexSpaceNode* child =
                part->get_child(part->color_space->linearize_color(*itr));
            if (child->set_domain(
                    Domain(lo, hi), ApEvent::NO_AP_EVENT,
                    false /*take ownership*/, true /*broadcast*/))
              delete child;
          }
        }
      }
      if (done)
      {
        if (runtime->safe_model)
          validate_output_extents(index);
        IndexSpaceNode* parent =
            runtime->get_node(output_regions[index].parent.get_index_space());
        // If we're done we can set the parent index space result
        if (options.global_indexing())
        {
          DomainPoint lo, hi;
          lo.dim = color.dim;
          hi.dim = color.dim;
          for (int dim = 0; dim < color.dim; dim++)
          {
            lo[dim] = 0;
            hi[dim] = state.offsets[dim].back() - 1;  // inclusive
          }
          if (parent->set_domain(
                  Domain(lo, hi), ApEvent::NO_AP_EVENT,
                  false /*take ownership*/, true /*broadcast*/))
            delete parent;
        }
        else if (parent->set_output_union(state.points))
          delete parent;
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::validate_output_extents(unsigned index)
    //--------------------------------------------------------------------------
    {
      legion_assert(output_regions.size() == output_region_states.size());
      legion_assert(output_regions.size() == output_region_options.size());
      const size_t expected_points = launch_space->get_volume();
      if (output_region_options[index].bounded_requirement())
        return;
      if (output_region_states[index].points.size() == expected_points)
        return;
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error << "A projection functor for every output requirement must be "
            << "bijective, but projection functor "
            << output_regions[index].projection << " for output requirement "
            << index << " in task " << *this << " mapped more than one point "
            << "in the launch domain to the same subregion.";
      error.raise();
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_output_registered(RtEvent registered)
    //--------------------------------------------------------------------------
    {
      legion_assert(registered.exists());
      legion_assert(!output_regions.empty());
      // Record it in the set of output events and if we've seen all of them
      // then we can launch the meta-task to do the final regisration
      AutoLock o_lock(op_lock);
      output_preconditions.emplace_back(registered);
      legion_assert(output_preconditions.size() <= total_points);
      if (output_preconditions.size() == total_points)
      {
        // Can only mark the EqKDTree ready once all the points are registered
        FinalizeOutputEqKDTreeArgs args(this);
        registered = runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY,
            Runtime::merge_events(output_preconditions));
        commit_preconditions.insert(registered);
      }
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::initialize_task(
        InnerContext* ctx, const IndexTaskLauncher& launcher,
        IndexSpace launch_sp, Provenance* provenance, bool track /*= true*/,
        std::vector<OutputRequirement>* outputs /*= nullptr*/)
    //--------------------------------------------------------------------------
    {
      parent_ctx = ctx;
      task_id = launcher.task_id;
      indexes = launcher.index_requirements;
      initialize_regions(launcher.region_requirements);
      futures = launcher.futures;
      // If the task has any output requirements, we create fresh region and
      // partition names and return them back to the user
      if (outputs != nullptr)
        create_output_regions(*outputs, launch_sp);
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);

      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
        arg_manager.save_buffer(launcher.global_arg.get_ptr(), arglen);
        args = arg_manager.get_buffer();
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
      concurrent_functor = launcher.concurrent_functor;
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      is_index_space = true;
      legion_assert(launch_sp.exists());
      legion_assert(launch_space == nullptr);
      launch_space = runtime->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        index_domain = launch_space->get_tight_domain();
      else
        index_domain = launcher.launch_domain;
      internal_space = launch_space->handle;
      sharding_space = launcher.sharding_space;
      initialize_base_task(ctx, launcher.predicate, task_id, provenance);
      if (outputs != nullptr)
      {
        if (launcher.predicate != Predicate::TRUE_PRED)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Output requirements are disallowed for tasks launched with "
                   "predicates, "
                << "but predicated task launch for task " << get_task_name()
                << " (" << get_unique_id() << ") in parent task "
                << parent_ctx->get_task_name() << " (UID "
                << parent_ctx->get_unique_id()
                << ") is used with output requirements.";
          error.raise();
        }
        if (get_trace() != nullptr)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Output requirements are disallowed for tasks launched "
                   "inside traces. "
                << "Task " << get_task_name() << " (UID " << get_unique_id()
                << ") in parent task " << parent_ctx->get_task_name()
                << " (UID " << parent_ctx->get_unique_id()
                << ") has output requirements in trace "
                << get_trace()->get_trace_id() << ".";
          error.raise();
        }
      }
      if (!launcher.elide_future_return)
      {
        if (launcher.predicate != Predicate::TRUE_PRED)
          initialize_predicate(
              launcher.predicate_false_future, launcher.predicate_false_result);
        future_map = create_future_map(
            ctx, launch_space->handle, launcher.sharding_space);
        future_return_size = launcher.future_return_size;
      }
      else
        elide_future_return = true;
      // Make sure you do this after making the output regions
      compute_parent_indexes(false /*force*/);
      if (concurrent_task && parent_ctx->is_concurrent_context())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Illegal nested concurrent index space task launch "
            << get_task_name() << " (UID " << get_unique_id()
            << ") inside task " << parent_ctx->get_task_name() << " (UID "
            << parent_ctx->get_unique_id()
            << ") which has a concurrent ancestor (must epoch or index task). "
            << "Nested concurrency is not supported.";
        error.raise();
      }
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        // Don't log this yet if we're part of a must epoch operation
        if (track)
          LegionSpy::log_index_task(
              parent_ctx->get_unique_id(), unique_op_id, task_id,
              get_task_name());
        for (const PhaseBarrier& barrier : launcher.wait_barriers)
        {
          ApEvent e = Runtime::get_previous_phase(barrier.phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
      }
      return future_map;
    }

    //--------------------------------------------------------------------------
    Future IndexTask::initialize_task(
        InnerContext* ctx, const IndexTaskLauncher& launcher,
        IndexSpace launch_sp, Provenance* provenance, ReductionOpID redop_id,
        bool deterministic, bool track /*= true*/,
        std::vector<OutputRequirement>* outputs /*= nullptr*/)
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
      if (outputs != nullptr)
        create_output_regions(*outputs, launch_sp);
      update_grants(launcher.grants);
      wait_barriers = launcher.wait_barriers;
      update_arrival_barriers(launcher.arrive_barriers);

      arglen = launcher.global_arg.get_size();
      if (arglen > 0)
      {
        arg_manager.save_buffer(launcher.global_arg.get_ptr(), arglen);
        args = arg_manager.get_buffer();
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
      concurrent_functor = launcher.concurrent_functor;
      map_id = launcher.map_id;
      tag = launcher.tag;
      mapper_data_size = launcher.map_arg.get_size();
      if (mapper_data_size > 0)
      {
        legion_assert(mapper_data == nullptr);
        mapper_data = malloc(mapper_data_size);
        memcpy(mapper_data, launcher.map_arg.get_ptr(), mapper_data_size);
      }
      is_index_space = true;
      legion_assert(launch_sp.exists());
      legion_assert(launch_space == nullptr);
      launch_space = runtime->get_node(launch_sp);
      add_launch_space_reference(launch_space);
      if (!launcher.launch_domain.exists())
        index_domain = launch_space->get_tight_domain();
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
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Reduction operation " << redop << " for index task launch "
              << get_task_name() << " (ID " << get_unique_id()
              << ") is not foldable.";
        error.raise();
      }
      initialize_base_task(ctx, launcher.predicate, task_id, provenance);
      if (outputs != nullptr)
      {
        if (launcher.predicate != Predicate::TRUE_PRED)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Output requirements are disallowed for tasks launched with "
                   "predicates, "
                << "but predicated task launch for task " << get_task_name()
                << " (" << get_unique_id() << ") in parent task "
                << parent_ctx->get_task_name() << " (UID "
                << parent_ctx->get_unique_id()
                << ") is used with output requirements.";
          error.raise();
        }
        if (get_trace() != nullptr)
        {
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "Output requirements are disallowed for tasks launched "
                   "inside traces. "
                << "Task " << get_task_name() << " (UID " << get_unique_id()
                << ") in parent task " << parent_ctx->get_task_name()
                << " (UID " << parent_ctx->get_unique_id()
                << ") has output requirements in trace "
                << get_trace()->get_trace_id() << ".";
          error.raise();
        }
      }
      if (launcher.predicate != Predicate::TRUE_PRED)
        initialize_predicate(
            launcher.predicate_false_future, launcher.predicate_false_result);
      reduction_future = Future(new FutureImpl(
          parent_ctx, true /*register*/,
          runtime->get_available_distributed_id(), provenance, this));
      if (serdez_redop_fns == nullptr)
      {
        reduction_future_size = reduction_op->sizeof_rhs;
        reduction_future.impl->set_future_result_size(
            reduction_op->sizeof_rhs, runtime->address_space);
      }
      else if (launcher.future_return_size)
      {
        reduction_future_size = launcher.future_return_size;
        reduction_future.impl->set_future_result_size(
            *reduction_future_size, runtime->address_space);
      }
      // Make sure you do this after making the output regions
      compute_parent_indexes(false /*force*/);
      if (concurrent_task && parent_ctx->is_concurrent_context())
      {
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error
            << "Illegal nested concurrent index space task launch "
            << get_task_name() << " (UID " << get_unique_id()
            << ") inside task " << parent_ctx->get_task_name() << " (UID "
            << parent_ctx->get_unique_id()
            << ") which has a concurrent ancestor (must epoch or index task). "
            << "Nested concurrency is not supported.";
        error.raise();
      }
      if ((spy_logging_level > NO_SPY_LOGGING) && track)
      {
        LegionSpy::log_index_task(
            parent_ctx->get_unique_id(), unique_op_id, task_id,
            get_task_name());
        for (const PhaseBarrier& barrier : launcher.wait_barriers)
        {
          ApEvent e = Runtime::get_previous_phase(barrier.phase_barrier);
          LegionSpy::log_phase_barrier_wait(unique_op_id, e);
        }
        LegionSpy::log_future_creation(
            unique_op_id, reduction_future.impl->did, index_point);
      }
      return reduction_future;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_regions(const std::vector<RegionRequirement>& rs)
    //--------------------------------------------------------------------------
    {
      regions = rs;
      // Rewrite any singular region requirements to projections
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        RegionRequirement& req = regions[idx];
        if (req.handle_type == LEGION_SINGULAR_PROJECTION)
        {
          req.handle_type = LEGION_REGION_PROJECTION;
          req.projection = 0;  // identity
        }
        // These are some checks for sanity if the user is using the default
        // projection functor from an upper bound region to make sure they
        // know what they are doing
        if (IS_WRITE(req) && (req.projection == 0) &&
            (req.handle_type == LEGION_REGION_PROJECTION) &&
            runtime->runtime_warnings)
        {
          Warning warning;
          warning << "Parent task " << *parent_ctx
                  << " issued index space task " << *this
                  << " with non-scalable projection region requirement " << idx
                  << "that ensures all point tasks will be reading and "
                     "writing to "
                  << "the same logical region. This implies there will be no "
                     "task "
                  << "parallelism in this index space task launch.";
          warning.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_predicate(
        const Future& pred_future, const UntypedBuffer& pred_arg)
    //--------------------------------------------------------------------------
    {
      if (pred_future.impl != nullptr)
        predicate_false_future = pred_future;
      else if (pred_arg.get_size() > 0)
        predicate_false_result.save_buffer(
            pred_arg.get_ptr(), pred_arg.get_size());
    }

    //--------------------------------------------------------------------------
    void IndexTask::prepare_map_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch != nullptr);
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
      // Initialize the privilege paths
      if (!options_selected)
      {
        const bool inline_task = select_task_options(false /*prioritize*/);
        if (inline_task)
        {
          Warning warning;
          warning
              << "Mapper " << *mapper << " requested to inline task " << *this
              << " but the 'enable_inlining' option was "
              << "not set on the task launcher so the request is being ignored";
          warning.raise();
        }
      }
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        for (unsigned idx = 0; idx < logical_regions.size(); idx++)
          TaskOp::log_requirement(unique_op_id, idx, logical_regions[idx]);
        log_launch_space(launch_space->handle);
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
        std::vector<OutputRequirement>& outputs, IndexSpace launch_space)
    //--------------------------------------------------------------------------
    {
      size_t num_tasks = runtime->get_domain_volume(launch_space);
      Provenance* provenance = get_provenance();
      output_region_options.resize(outputs.size());
      output_region_states.resize(outputs.size());
      bool has_global_indexing = false;
      for (unsigned idx = 0; idx < outputs.size(); idx++)
      {
        OutputRequirement& req = outputs[idx];
        output_region_options[idx] = OutputOptions(
            req.global_indexing, req.bounded_requirement, false /*grouped*/);

        IndexSpaceNode* color_space = req.color_space.exists() ?
                                          runtime->get_node(req.color_space) :
                                          runtime->get_node(launch_space);
        const unsigned color_dim = color_space->get_num_dims();
        if (!req.bounded_requirement)
        {
          TypeTag type_tag;
          const unsigned requested_dim =
              Internal::NT_TemplateHelper::get_dim(req.type_tag);
          if (req.global_indexing)
          {
            has_global_indexing = true;
            if (color_dim != requested_dim)
            {
              Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              error << "Output region " << idx << " of task " << *this
                    << " is requested to have " << requested_dim
                    << " dimensions, but the color space has "
                    << index_domain.get_dim()
                    << " dimensions. Dimensionalities "
                    << "of output regions must be the same as the "
                    << "color space's in global indexing mode.";
              error.raise();
            }
            if (req.global_indexing && color_space->is_sparse())
            {
              Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              error << "The global indexing mode requires the color space of "
                    << "an output requirement to be dense, but a sparse color "
                    << "space is assigned to output requirement " << idx
                    << " of task " << *this << ".";
              error.raise();
            }
            if (color_space->get_volume() != num_tasks)
            {
              Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              error
                  << "Output region " << idx << " of task " << *this
                  << " requests projection but the volume of the color space "
                  << "is different from the total number of point tasks. "
                  << "The mapping between the launch domain and the subregions "
                  << "must be bijective.";
              error.raise();
            }

            type_tag = req.type_tag;
            OutputRegionState& state = output_region_states[idx];
            state.extents.resize(color_dim);
            state.offsets.resize(color_dim);
            const Domain color_domain = color_space->get_tight_domain();
            state.global_color_space = color_domain;
            state.projection =
                runtime->find_projection_function(req.projection);
            if ((req.projection != 0) && !state.projection->is_invertible)
            {
              Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
              error << "Output region " << idx << " of task " << *this
                    << " is using custom projection function " << req.projection
                    << " for global indexing but that projection function "
                    << "is not invertible. All custom projection functions "
                    << "used with global indexing for output regions must "
                    << "be invertible.";
              error.raise();
            }
            // All offsets start at zero
            for (unsigned dim = 0; dim < color_dim; dim++)
            {
              const size_t color_dim_size =
                  (color_domain.hi()[dim] - color_domain.lo()[dim]) + 1;
              legion_assert(color_dim_size > 0);
              state.extents[dim].resize(color_dim_size, -1);
              state.offsets[dim].resize(color_dim_size + 1, -1);
              state.offsets[dim].front() = 0;
            }
          }
          else
          {
            // When local indexing is used for the output region,
            // we create an (N+1)-D index space when the color domain is N-D.

            // Before creating the index space, we make sure that
            // the dimensionality (N+1) does not exceed LEGION_MAX_DIM.
            if ((color_dim + requested_dim) > LEGION_MAX_DIM)
            {
              Error error(LEGION_INTERFACE_EXCEPTION);
              error
                  << "Dimensionality of output region " << idx << " of task "
                  << *this << " exceeded LEGION_MAX_DIM=" << LEGION_MAX_DIM
                  << ". You may rebuild your code with a bigger LEGION_MAX_DIM "
                  << "value or reduce dimensionality of either the color space "
                  << "or the output region.";
              error.raise();
            }

            OutputRegionTagCreator creator(&type_tag, color_dim);
            Internal::NT_TemplateHelper::demux<OutputRegionTagCreator>(
                req.type_tag, &creator);
          }

          // Create a deferred index space
          IndexSpace index_space =
              parent_ctx->create_unbound_index_space(type_tag, provenance);

          // Create a pending partition using the launch domain as the color
          // space
          IndexPartition pid = parent_ctx->create_pending_partition(
              index_space, color_space->handle, LEGION_DISJOINT_COMPLETE_KIND,
              LEGION_AUTO_GENERATE_ID, provenance, true /*trust partitioning*/);

          // Instantiate all local children to ensure that others can refer
          // to them even before they've been computed if necessary
          IndexPartNode* index_part = runtime->get_node(pid);
          for (ColorSpaceIterator itr(index_part, true /*local*/); itr; itr++)
          {
            // This will force the instantiation of the child without blocking
            RtEvent instantiated;
            index_part->get_child(*itr, &instantiated);
            if (instantiated.exists())
              commit_preconditions.insert(instantiated);
          }

          // Create an output region and a partition
          LogicalRegion region = parent_ctx->create_logical_region(
              index_space, req.field_space, false /*local region*/, provenance,
              true /*output region*/);

          LogicalPartition partition =
              runtime->get_logical_partition(region, pid);

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
      if (has_global_indexing)
        output_pointwise_dependences.resize(outputs.size());
    }

    //--------------------------------------------------------------------------
    void IndexTask::perform_base_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // To be correct with the new scheduler we also have to
      // register mapping dependences on futures
      for (const Future& future : futures)
        if (future.impl != nullptr)
          future.impl->register_dependence(this);
      if (predicate_false_future.impl != nullptr)
        predicate_false_future.impl->register_dependence(this);
      // Always have to register a full dependence on this since we need
      // to have the producer mapped by the time we're enumerating points
      if (point_arguments.impl != nullptr)
        point_arguments.impl->register_dependence(this);
      // Register mapping dependences on any future maps also
      // if we're not pointwise analyzable. If we are then we'll
      // do pointwise analysis on these when we enumerate the points
      if (!is_pointwise_analyzable())
      {
        for (const FutureMap& future_map : point_futures)
          future_map.impl->register_dependence(this);
      }
      else if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        // Record pointwise dependences on the point futures
        for (const FutureMap& future_map : point_futures)
        {
          if (future_map.impl->op == nullptr)
            continue;
          LegionSpy::log_future_dependence(
              parent_ctx->get_unique_id(), future_map.impl->op_uid,
              unique_op_id, true /*pointwise*/);
        }
      }
      if (!wait_barriers.empty() || !arrive_barriers.empty())
        parent_ctx->perform_barrier_dependence_analysis(
            this, wait_barriers, arrive_barriers, must_epoch);
      version_infos.resize(logical_regions.size());
    }

    //--------------------------------------------------------------------------
    void IndexTask::report_interfering_requirements(
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
      interfering_requirements.insert(
          std::pair<unsigned, unsigned>(idx1, idx2));
    }

    //--------------------------------------------------------------------------
    bool IndexTask::record_trace_hash(
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
      hasher.hash(concurrent_functor);
      hasher.hash<bool>(concurrent_task);
      hasher.hash<bool>(must_epoch_task);
      hasher.hash(index_domain);
      if (future_return_size)
        hasher.hash(*future_return_size);
      if (reduction_future_size)
        hasher.hash(*reduction_future_size);
      return recorder.record_operation_hash(this, hasher, opidx);
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      if (runtime->safe_model)
        start_check_point_requirements();
      // Do a quick test for empty index space launches
      total_points = launch_space->get_volume();
      if (total_points == 0)
      {
        // Clean up this task execution if there are no points
        complete_mapping();
        complete_execution();
        trigger_children_committed();
      }
      else
      {
        // If we had a trace then we might need to recompute our parent
        // region indexes now since we might have skipped it earlier
        if (trace != nullptr)
          compute_parent_indexes(true /*force*/);
        // Enumerate the futures in the future map
        if ((redop == 0) && !elide_future_return)
          enumerate_futures(index_domain);
        if (concurrent_task)
        {
          // Instantiate preconditions for each of the colors
          ConcurrentColoringFunctor* functor =
              runtime->find_concurrent_coloring_functor(concurrent_functor);
          if (functor->supports_max_color())
          {
            Color max_color = functor->max_color(index_domain);
            for (Color color = 0; color <= max_color; color++)
              concurrent_groups[color].precondition.interpreted =
                  Runtime::create_rt_user_event();
          }
          else
          {
            // If we're recording we need to iterate all the points to
            // count how many points are associated with each color so
            // we can make the barrier to be used later
            for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
            {
              Color color = functor->color(*itr, index_domain);
              std::map<Color, ConcurrentGroup>::iterator finder =
                  concurrent_groups.find(color);
              if (finder == concurrent_groups.end())
              {
                finder = concurrent_groups
                             .emplace(std::make_pair(color, ConcurrentGroup()))
                             .first;
                finder->second.precondition.interpreted =
                    Runtime::create_rt_user_event();
              }
            }
          }
        }
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
    void IndexTask::enumerate_futures(const Domain& domain)
    //--------------------------------------------------------------------------
    {
      legion_assert(!elide_future_return);
      legion_assert(future_handles == nullptr);
      future_handles = new FutureHandles;
      future_handles->add_reference();
      std::map<DomainPoint, PackedFutureHandle>& handles =
          future_handles->handles;
      for (Domain::DomainPointIterator itr(domain); itr; itr++)
      {
        Future f = future_map.impl->get_future(itr.p, true /*internal only*/);
        PackedFutureHandle& handle = handles[itr.p];
        handle.future_did = f.impl->did;
        // Pack a reference that will be unpacked once the future is set
        f.impl->pack_global_ref(handle.lamport_clock);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::rendezvous_concurrent_mapped(
        const DomainPoint& point, Processor target, Color color,
        RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      legion_assert(concurrent_task);
      bool done = false;
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
      legion_assert(finder != concurrent_groups.end());
      {
        AutoLock o_lock(op_lock);
        if (precondition.exists())
          finder->second.preconditions.emplace_back(precondition);
        std::map<Processor, DomainPoint>::const_iterator proc_finder =
            finder->second.processors.find(target);
        if (proc_finder != finder->second.processors.end())
          report_concurrent_mapping_failure(target, point, proc_finder->second);
        finder->second.processors[target] = point;
        finder->second.group_points++;
        legion_assert(concurrent_points < total_points);
        done = (++concurrent_points == total_points);
      }
      if (done)
        finalize_concurrent_mapped();
    }

    //--------------------------------------------------------------------------
    void IndexTask::rendezvous_concurrent_mapped(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      legion_assert(concurrent_task);
      bool done = false;
      {
        size_t num_colors;
        derez.deserialize(num_colors);
        AutoLock o_lock(op_lock);
        for (unsigned idx1 = 0; idx1 < num_colors; idx1++)
        {
          Color color;
          derez.deserialize(color);
          std::map<Color, ConcurrentGroup>::iterator finder =
              concurrent_groups.find(color);
          legion_assert(finder != concurrent_groups.end());
          RtEvent precondition;
          derez.deserialize(precondition);
          if (precondition.exists())
            finder->second.preconditions.emplace_back(precondition);
          size_t num_points;
          derez.deserialize(num_points);
          for (unsigned idx2 = 0; idx2 < num_points; idx2++)
          {
            Processor target;
            derez.deserialize(target);
            DomainPoint point;
            derez.deserialize(point);
            std::map<Processor, DomainPoint>::const_iterator proc_finder =
                finder->second.processors.find(target);
            if (proc_finder != finder->second.processors.end())
              report_concurrent_mapping_failure(
                  target, point, proc_finder->second);
            finder->second.processors[target] = point;
          }
          finder->second.group_points += num_points;
          concurrent_points += num_points;
          legion_assert(concurrent_points <= total_points);
        }
        done = (concurrent_points == total_points);
      }
      if (done)
        finalize_concurrent_mapped();
    }

    //--------------------------------------------------------------------------
    void IndexTask::finalize_concurrent_mapped(void)
    //--------------------------------------------------------------------------
    {
      for (std::pair<const Color, ConcurrentGroup>& group : concurrent_groups)
      {
        legion_assert(group.second.precondition.interpreted.exists());
        group.second.color_points = group.second.group_points;
        if (is_recording())
        {
          // Create and record the barrier with the right number of arrivals
          // for this concurrent group for later iterations
          const RtBarrier barrier =
              runtime->create_rt_barrier(group.second.color_points);
          group.second.shards.resize(1, 0);
          tpl->record_concurrent_group(
              this, group.first, group.second.group_points,
              group.second.color_points, barrier, group.second.shards);
        }
        if (!group.second.preconditions.empty())
          Runtime::trigger_event(
              group.second.precondition.interpreted,
              Runtime::merge_events(group.second.preconditions));
        else
          Runtime::trigger_event(group.second.precondition.interpreted);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_concurrent_group(
        Color color, size_t local, size_t global, RtBarrier barrier,
        const std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(concurrent_groups.find(color) == concurrent_groups.end());
      ConcurrentGroup& group = concurrent_groups[color];
      group.group_points = local;
      group.color_points = global;
      group.precondition.traced = barrier;
    }

    //--------------------------------------------------------------------------
    void IndexTask::initialize_must_epoch_concurrent_group(
        Color color, RtUserEvent precondition)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch_task);
      legion_assert(concurrent_groups.find(color) == concurrent_groups.end());
      ConcurrentGroup& group = concurrent_groups[color];
      group.precondition.interpreted = precondition;
    }

    //--------------------------------------------------------------------------
    void IndexTask::concurrent_allreduce(
        Color color, SliceTask* slice, AddressSpaceID slice_space,
        size_t points, uint64_t lamport_clock, VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      if (must_epoch_task)
      {
        legion_assert(color == 0);
        must_epoch->concurrent_allreduce(
            slice, slice_space, points, lamport_clock, poisoned);
        return;
      }
      bool done = false;
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
      legion_assert(finder != concurrent_groups.end());
      {
        AutoLock o_lock(op_lock);
        if (finder->second.lamport_clock < lamport_clock)
          finder->second.lamport_clock = lamport_clock;
        if (poisoned)
          finder->second.poisoned = true;
        if (finder->second.slice_tasks.empty())
          finder->second.variant = vid;
        else if (finder->second.variant != vid)
          finder->second.variant = std::min(finder->second.variant, vid);
        finder->second.slice_tasks.emplace_back(
            std::make_pair(slice, slice_space));
        legion_assert(points <= finder->second.group_points);
        finder->second.group_points -= points;
        done = (finder->second.group_points == 0);
      }
      if (done)
      {
        if (finder->second.variant > 0)
        {
          // Check to see if this variant needs a task barrier
          VariantImpl* variant =
              runtime->find_variant_impl(task_id, finder->second.variant);
          if (variant->needs_barrier())
            finder->second.task_barrier =
                runtime->create_rt_barrier(finder->second.color_points);
        }
        // Swap this vector onto the stack in case the slice task gets deleted
        // out from under us while we are finalizing things
        std::vector<std::pair<SliceTask*, AddressSpaceID>> local_copy;
        local_copy.swap(finder->second.slice_tasks);
        for (const std::pair<SliceTask*, AddressSpaceID>& slice_pair :
             local_copy)
        {
          if (slice_pair.second != runtime->address_space)
          {
            SliceConcurrentResponse rez;
            {
              RezCheck z(rez);
              rez.serialize(slice_pair.first);
              rez.serialize(color);
              rez.serialize(finder->second.task_barrier);
              rez.serialize(finder->second.lamport_clock);
              rez.serialize(finder->second.variant);
              rez.serialize(finder->second.poisoned);
            }
            rez.dispatch(slice_pair.second);
          }
          else
            slice_pair.first->finish_concurrent_allreduce(
                color, finder->second.lamport_clock, finder->second.poisoned,
                finder->second.variant, finder->second.task_barrier);
        }
      }
    }

    //--------------------------------------------------------------------------
    uint64_t IndexTask::collective_lamport_allreduce(
        uint64_t lamport_clock, size_t points, bool need_result)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      if (collective_lamport_clock < lamport_clock)
        collective_lamport_clock = lamport_clock;
      collective_unbounded_points += points;
      legion_assert(collective_unbounded_points <= total_points);
      if (collective_unbounded_points < total_points)
      {
        if (need_result)
        {
          if (!collective_lamport_clock_ready.exists())
            collective_lamport_clock_ready = Runtime::create_rt_user_event();
          o_lock.release();
          collective_lamport_clock_ready.wait();
        }
      }
      else if (collective_lamport_clock_ready.exists())
        Runtime::trigger_event(collective_lamport_clock_ready);
      return collective_lamport_clock;
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
            runtime->find_domain(internal_space, local_domain);
            // Handling the future map case
            if (predicate_false_future.impl != nullptr)
            {
              for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
              {
                Future f =
                    future_map.impl->get_future(itr.p, true /*internal*/);
                // Safe to block indefinitely waiting for unbounded pools
                f.impl->set_result(
                    this, predicate_false_future.impl,
                    nullptr /*safe_for_unbounded_pools*/);
              }
            }
            else
            {
              for (Domain::DomainPointIterator itr(local_domain); itr; itr++)
              {
                Future f =
                    future_map.impl->get_future(itr.p, true /*internal*/);
                if (predicate_false_result.get_size() > 0)
                  f.impl->set_local(
                      predicate_false_result.get_buffer(),
                      predicate_false_result.get_size(), false /*own*/);
                else
                  f.impl->set_result(ApEvent::NO_AP_EVENT, nullptr);
              }
            }
          }
        }
        else
        {
          // Handling a reduction case
          if (redop_initial_value.impl != nullptr)
          {
            // Safe to block here indefinitely waiting for unbounded pools
            reduction_future.impl->set_result(
                this, redop_initial_value.impl,
                nullptr /*safe_for_unbounded_pools*/);
          }
          else
            reduction_future.impl->set_local(
                &reduction_op->identity, reduction_op->sizeof_rhs);
        }
      }
      // Can check this without the lock since we know the predication state
      // has been marked correctly while holding the lock
      if (!pending_pointwise_dependences.empty())
      {
        // Just trigger these since the points won't be mapped anyway
        for (const std::pair<const DomainPoint, RtUserEvent>& dep :
             pending_pointwise_dependences)
          Runtime::trigger_event(dep.second);
        pending_pointwise_dependences.clear();
      }
      // Then clean up this task execution
      complete_mapping();
      complete_execution(execution_condition);
      trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::premap_task(void)
    //--------------------------------------------------------------------------
    {
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
      if (mapper == nullptr)
        mapper = runtime->find_mapper(current_proc, map_id);
      mapper->invoke_premap_task(this, input, output);
      // See if we need to update the new target processor
      if (output.new_target_proc.exists())
        this->target_proc = output.new_target_proc;
      create_future_instances(output.reduction_futures);
      // If we're recording this trace then we need to remember this
      if (is_recording())
      {
        legion_assert((tpl != nullptr) && tpl->is_recording());
        tpl->record_premap_output(this, output, map_applied_conditions);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::create_future_instances(std::vector<Memory>& target_mems)
    //--------------------------------------------------------------------------
    {
      legion_assert(reduction_instances.empty());
      if (!target_mems.empty())
      {
        if (target_mems.size() > 1)
        {
          std::set<Memory> unique_mems;
          for (std::vector<Memory>::iterator it = target_mems.begin();
               it != target_mems.end();
               /*nothing*/)
          {
            if (!it->exists())
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Invalid mapper output. Mapper " << *mapper
                    << " requested index task reduction future be mapped to a "
                    << "NO_MEMORY for task " << *this
                    << " which is illegal. All "
                    << "requests for mapping output futures must be mapped to "
                    << "actual memories.";
              error.raise();
            }
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
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Invalid mapper output. Mapper " << *mapper
                << " requested index task reduction future be mapped to a "
                << "NO_MEMORY for task " << *this << " which is illegal. "
                << "All requests for mapping output futures must be mapped "
                << "to actual memories.";
          error.raise();
        }
      }
      else
        target_mems.emplace_back(runtime->runtime_system_memory);
      // If we've got a serdez redop function then we don't know how big
      // the output is going to be until later, otherwise we know the
      // output size from the reduction operator
      if (serdez_redop_fns == nullptr)
      {
        reduction_instances.reserve(target_mems.size());
        TaskTreeCoordinates coordinates;
        compute_task_tree_coordinates(coordinates);
        int runtime_visible_index = -1;
        for (const Memory& memory : target_mems)
        {
          if ((runtime_visible_index < 0) &&
              FutureInstance::check_meta_visible(memory))
            runtime_visible_index = reduction_instances.size();
          MemoryManager* manager = runtime->find_memory_manager(memory);
          // Safe to block here indefinitely waiting for unbounded pools
          reduction_instances.emplace_back(manager->create_future_instance(
              unique_op_id, coordinates, reduction_op->sizeof_rhs,
              nullptr /*safe_for_unbounded_pools*/));
        }
        // This is an important optimization: if we're doing a small
        // reduction value we always want the reduction instance to
        // be somewhere meta visible for performance reasons, so we
        // make a meta-visible instance if we don't have one
        if ((runtime_visible_index < 0) &&
            (reduction_op->sizeof_rhs <= LEGION_MAX_RETURN_SIZE))
        {
          runtime_visible_index = reduction_instances.size();
          MemoryManager* manager =
              runtime->find_memory_manager(runtime->runtime_system_memory);
          // Safe to block here indefinitely waiting for unbounded pools
          reduction_instances.emplace_back(manager->create_future_instance(
              unique_op_id, coordinates, reduction_op->sizeof_rhs,
              nullptr /*safe_for_unbounded_pools*/));
        }
        if (runtime_visible_index > 0)
          std::swap(
              reduction_instances.front(),
              reduction_instances[runtime_visible_index]);
        legion_assert(reduction_instance == nullptr);
        reduction_instance = reduction_instances.front();
        // Need to initialize this with the reduction value
        if ((redop_initial_value.impl != nullptr) &&
            (parent_ctx->get_task()->get_shard_id() == 0))
          reduction_instance_precondition = redop_initial_value.impl->copy_to(
              reduction_instance, this, execution_fence_event);
        else
          reduction_instance_precondition =
              reduction_instance.load()->initialize(
                  reduction_op, this, execution_fence_event);
      }
      else
      {
        if ((redop_initial_value.impl != nullptr) &&
            (parent_ctx->get_task()->get_shard_id() == 0))
        {
          redop_initial_value.impl->request_runtime_instance(this);
          const void* value = redop_initial_value.impl->find_runtime_buffer(
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
      if (is_origin_mapped())
      {
        // This will only get called if we had slices that couldn't map, but
        // they have now all mapped
        legion_assert(slices.empty());
        // We're never actually run
        return false;
      }
      else
      {
        if (!is_sliced() && target_proc.exists() &&
            !runtime->is_local(target_proc))
        {
          // Make a slice copy and send it away
          SliceTask* clone = clone_as_slice_task(
              internal_space, target_proc, true /*needs slice*/, stealable);
          runtime->send_task(clone);
          return false;  // We have now been sent away
        }
        else
        {
          set_current_proc(target_proc);
          return true;  // Still local so we can be sliced
        }
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::perform_mapping(
        MustEpochOp* owner /*=nullptr*/,
        const DeferMappingArgs* args /*=nullptr*/)
    //--------------------------------------------------------------------------
    {
      // This will only get called if we had slices that failed to origin map
      legion_assert(!slices.empty());
      // Should never get duplicate invocations here
      legion_assert(args == nullptr);
      while (!slices.empty())
      {
        SliceTask* slice = slices.front();
        slices.pop_front();
        slice->trigger_mapping();
      }
      return false;
    }

    //--------------------------------------------------------------------------
    void IndexTask::launch_task(bool inline_task)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
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
    TaskOp::TaskKind IndexTask::get_task_kind(void) const
    //--------------------------------------------------------------------------
    {
      return INDEX_TASK_KIND;
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_operation_events(
          unique_op_id, ApEvent::NO_AP_EVENT, effects);
      // Set the future if we actually ran the task or we speculated
      if ((redop > 0) && (predication_state != PREDICATED_FALSE_STATE))
      {
        if (reduction_future_size &&
            (*reduction_future_size < reduction_instances.front()->size))
        {
          // This failure mode should only happen with serdez redops fns
          // since we should do the other reductions correctly ourself
          legion_assert(serdez_redop_fns != nullptr);
          Provenance* provenance = get_provenance();
          if (provenance != nullptr)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Index Task " << get_task_name() << " (UID "
                  << get_unique_id() << ", provenance: " << provenance->human
                  << ") produced a reduced "
                  << "future value of " << reduction_instances.front()->size
                  << " bytes which is larger than the dynamically specified "
                     "bounds "
                  << "of " << *reduction_future_size << " bytes.";
            error.raise();
          }
          else
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Index Task " << get_task_name() << " (UID "
                  << get_unique_id() << ") produced a reduced future value of "
                  << reduction_instances.front()->size
                  << " bytes which is larger than "
                  << "the dynamically specified bounds of "
                  << *reduction_future_size << " bytes.";
            error.raise();
          }
        }
        reduction_future.impl->set_results(
            effects, reduction_instances, reduction_metadata,
            reduction_metasize);
        // Clear this since we no longer own the buffer
        reduction_metadata = nullptr;
        reduction_instances.clear();
      }
      if (must_epoch != nullptr)
      {
        must_epoch->notify_subop_complete(this, effects);
        complete_operation(effects);
      }
      else
        complete_operation(effects);
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      if (profiling_reported.exists())
      {
        if (outstanding_profiling_requests == 0)
        {
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
        else
          commit_preconditions.insert(profiling_reported);
      }
      if (must_epoch != nullptr)
      {
        RtEvent commit_precondition;
        if (!commit_preconditions.empty())
          commit_precondition = Runtime::merge_events(commit_preconditions);
        must_epoch->notify_subop_commit(this, commit_precondition);
        commit_operation(true /*deactivate*/, commit_precondition);
      }
      else
      {
        // Mark that this operation is now committed
        if (!commit_preconditions.empty())
          commit_operation(
              true /*deactivate*/, Runtime::merge_events(commit_preconditions));
        else
          commit_operation(true /*deactivate*/);
      }
    }

    //--------------------------------------------------------------------------
    bool IndexTask::pack_task(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool IndexTask::unpack_task(
        Deserializer& derez, Processor current, AddressSpaceID,
        std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void IndexTask::perform_inlining(
        VariantImpl* variant, const std::deque<InstanceSet>& parent_regions)
    //--------------------------------------------------------------------------
    {
      total_points = launch_space->get_volume();
      if ((redop == 0) && !elide_future_return)
        enumerate_futures(index_domain);
      SliceTask* slice = clone_as_slice_task(
          launch_space->handle, current_proc, false /*recurse*/,
          false /*stealable*/);
      slice->enumerate_points(true /*inlining*/);
      slice->perform_inlining(variant, parent_regions);
    }

    //--------------------------------------------------------------------------
    SliceTask* IndexTask::clone_as_slice_task(
        IndexSpace is, Processor p, bool recurse, bool stealable)
    //--------------------------------------------------------------------------
    {
      SliceTask* result = runtime->get_operation<SliceTask>();
      result->initialize_base_task(
          parent_ctx, Predicate::TRUE_PRED, this->task_id, get_provenance());
      result->clone_multi_from(this, is, p, recurse, stealable);
      result->index_owner = this;
      LegionSpy::log_index_slice(get_unique_id(), result->get_unique_id());
      if (implicit_profiler != nullptr)
        implicit_profiler->register_slice_owner(
            get_unique_op_id(), result->get_unique_op_id());
      return result;
    }

    //--------------------------------------------------------------------------
    void IndexTask::reduce_future(
        const DomainPoint& point, FutureInstance* inst, ApEvent effects)
    //--------------------------------------------------------------------------
    {
      legion_assert(reduction_op != nullptr);
      // If we're doing a deterministic reduction then we need to
      // buffer up these future values until we get all of them so
      // that we can fold them in a deterministic way
      if (deterministic_redop)
      {
        // Store it in our temporary futures for later
        AutoLock o_lock(op_lock);
        legion_assert(temporary_futures.find(point) == temporary_futures.end());
        temporary_futures[point] = std::make_pair(inst, effects);
      }
      else
      {
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

    //--------------------------------------------------------------------------
    void IndexTask::pack_profiling_requests(
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
    int IndexTask::add_copy_profiling_request(
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
    bool IndexTask::handle_profiling_response(
        const Realm::ProfilingResponse& response, const void* orig,
        size_t orig_length, LgEvent& fevent, bool& failed_alloc)
    //--------------------------------------------------------------------------
    {
      const OpProfilingResponse* task_prof =
          static_cast<const OpProfilingResponse*>(response.user_data());
      Realm::ProfilingMeasurements::OperationFinishEvent finish_event;
      if (response.get_measurement(finish_event))
        fevent = LgEvent(finish_event.finish_event);
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
      mapper->invoke_task_report_profiling(this, info);
      const int count = outstanding_profiling_reported.fetch_add(1) + 1;
      legion_assert(count <= outstanding_profiling_requests);
      if (count == outstanding_profiling_requests)
        Runtime::trigger_event(profiling_reported);
      // Always record these as part of profiling
      return true;
    }

    //--------------------------------------------------------------------------
    void IndexTask::handle_profiling_update(int count)
    //--------------------------------------------------------------------------
    {
      legion_assert(count > 0);
      legion_assert(!mapped_event.has_triggered());
      outstanding_profiling_requests.fetch_add(count);
    }

    //--------------------------------------------------------------------------
    void IndexTask::register_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      // should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    FutureMap IndexTask::create_future_map(
        TaskContext* ctx, IndexSpace launch_space, IndexSpace sharding_space)
    //--------------------------------------------------------------------------
    {
      FutureMapImpl* result = new FutureMapImpl(
          ctx, this, this->launch_space,
          runtime->get_available_distributed_id(), get_provenance());
      future_map_coordinate = result->blocking_index;
      return FutureMap(result);
    }

    //--------------------------------------------------------------------------
    RtEvent IndexTask::find_pointwise_dependence(
        const DomainPoint& point, GenerationID needed_gen, bool intra_space,
        RtUserEvent to_trigger, std::optional<unsigned> output_region)
    //--------------------------------------------------------------------------
    {
      legion_assert(!output_region);
      AutoLock o_lock(op_lock);
      legion_assert(needed_gen <= gen);
      if ((needed_gen < gen) || mapped ||
          (predication_state == PREDICATED_FALSE_STATE))
      {
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
        return RtEvent::NO_RT_EVENT;
      }
      RtEvent point_mapped;
      // See if we can find this in the point mapped events data structure
      std::map<DomainPoint, RtEvent>::const_iterator finder =
          point_mapped_events.find(point);
      if (finder == point_mapped_events.end())
      {
        // Create a pending pointwise dependence for this point
        std::map<DomainPoint, RtUserEvent>::const_iterator pending_finder =
            pending_pointwise_dependences.find(point);
        if (pending_finder == pending_pointwise_dependences.end())
        {
          RtUserEvent pending = Runtime::create_rt_user_event();
          pending_pointwise_dependences.emplace(std::make_pair(point, pending));
          point_mapped = pending;
        }
        else
          point_mapped = pending_finder->second;
      }
      else
        point_mapped = finder->second;
      // If this is an intra-space look-up or we don't have output regions
      // then we are already done
      if (intra_space || output_pointwise_dependences.empty())
      {
        if (to_trigger.exists())
        {
          Runtime::trigger_event(to_trigger, point_mapped);
          return to_trigger;
        }
        else
          return point_mapped;
      }
      // Now for the hairy part: if we have globally indexed output regions then
      // we also need to compute transitive dependences for each of the points
      // that this next point will depend on as well in order to run safely
      std::vector<RtEvent> preconditions(1, point_mapped);
      for (unsigned idx = 0; idx < output_pointwise_dependences.size(); idx++)
      {
        if (!output_region_options[idx].global_indexing())
          continue;
        std::map<DomainPoint, RtEvent>& output_pointwise =
            output_pointwise_dependences[idx];
        DomainPoint color_point = point;
        IndexPartNode* partition = nullptr;
        ProjectionFunction* projection = nullptr;
        const RegionRequirement& req = output_regions[idx];
        const OutputRegionState& state = output_region_states[idx];
        if (req.projection != 0)
        {
          // Need to convert the point into the color space to check if we
          // have it already
          partition = runtime->get_node(req.partition.get_index_partition());
          projection = runtime->find_projection_function(req.projection);
          LogicalRegion region = projection->project_point(
              this, regions.size() + idx, index_domain, point);
          legion_assert(region.exists());
          color_point = runtime->get_node(region.get_index_space())
                            ->get_domain_point_color();
          legion_assert(state.global_color_space.contains(color_point));
        }
        finder = output_pointwise.find(color_point);
        if (finder == output_pointwise.end())
        {
          // DFS backwards for this output region to find the points we need
          std::vector<DomainPoint> needed_colors(1, color_point);
          while (!needed_colors.empty())
          {
            const DomainPoint& needed_color = needed_colors.back();
            std::vector<RtEvent> point_preconditions;
            // Check to see if we have all our previous points
            bool has_all_previous = true;
            for (int dim = 0; dim < needed_color.get_dim(); dim++)
            {
              DomainPoint previous = needed_color;
              previous[dim]--;
              if (!state.global_color_space.contains(previous))
                continue;
              finder = output_pointwise.find(previous);
              if (finder == output_pointwise.end())
              {
                needed_colors.push_back(previous);
                has_all_previous = false;
                break;
              }
              point_preconditions.push_back(finder->second);
            }
            if (!has_all_previous)
              continue;
            // Also include the particular point's mapped event
            DomainPoint orig_point = needed_color;
            if (projection != nullptr)
            {
              // Need to convert from color space back to the index space
              IndexSpaceNode* child = partition->get_child(
                  partition->color_space->linearize_color(needed_color));
              LogicalRegion region(
                  req.partition.tree_did, child->handle,
                  req.partition.field_space);
              std::vector<DomainPoint> points;
              projection->functor->invert(
                  region, req.partition, index_domain, points);
              if (points.size() != 1)
              {
                Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
                error
                    << "Illegal projection function inversion computed by "
                    << "projection functor " << projection->projection_id
                    << " for output region requirement " << idx << " of "
                    << *this << ". The projection function says that "
                    << points.size()
                    << " different points map to logical region " << region
                    << ", however projection functions for output "
                    << "region requirements are required to be bijiective and "
                    << "and therefore exactly one point should ever be allowed "
                    << "to map to a single output logical region.";
                error.raise();
              }
              orig_point = points.back();
            }
            finder = point_mapped_events.find(orig_point);
            if (finder == point_mapped_events.end())
            {
              std::map<DomainPoint, RtUserEvent>::const_iterator
                  pending_finder =
                      pending_pointwise_dependences.find(orig_point);
              if (pending_finder == pending_pointwise_dependences.end())
              {
                RtUserEvent pending = Runtime::create_rt_user_event();
                pending_pointwise_dependences.emplace(
                    std::make_pair(orig_point, pending));
              }
              else
                point_preconditions.push_back(pending_finder->second);
            }
            else
              point_preconditions.push_back(finder->second);
            output_pointwise[needed_color] =
                Runtime::merge_events(point_preconditions);
            needed_colors.pop_back();
          }
          finder = output_pointwise.find(point);
          legion_assert(finder != output_pointwise.end());
        }
        preconditions.push_back(finder->second);
      }
      if (to_trigger.exists())
      {
        Runtime::trigger_event(
            to_trigger, Runtime::merge_events(preconditions));
        return to_trigger;
      }
      else
        return Runtime::merge_events(preconditions);
    }

    //--------------------------------------------------------------------------
    void IndexTask::record_origin_mapped_slice(SliceTask* local_slice)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      origin_mapped_slices.emplace_back(local_slice);
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_point_mapped(
        const DomainPoint& point, RtEvent mapped_event)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      bool trigger_children_commit = false;
      {
        AutoLock o_lock(op_lock);
        legion_assert(
            point_mapped_events.find(point) == point_mapped_events.end());
        point_mapped_events.emplace(std::make_pair(point, mapped_event));
        std::map<DomainPoint, RtUserEvent>::iterator finder =
            pending_pointwise_dependences.find(point);
        if (finder != pending_pointwise_dependences.end())
        {
          Runtime::trigger_event(finder->second, mapped_event);
          pending_pointwise_dependences.erase(finder);
        }
        if (mapped_event.exists())
          map_applied_conditions.insert(mapped_event);
        legion_assert(mapped_points < total_points);
        if (++mapped_points == total_points)
        {
          // Don't complete this yet if we have redop serdez fns because
          // we still need to map the output future instance before we
          // can consider ourselves mapped and we can't do that until we
          // get the final future value
          if (serdez_redop_fns == nullptr)
            need_trigger = true;
          if ((committed_points == total_points) && !children_commit_invoked)
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
      if (trigger_children_commit)
        trigger_children_committed();
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_complete(
        unsigned points, ApEvent slice_effect, void* metadata /*= nullptr*/,
        size_t metasize /*= 0*/)
    //--------------------------------------------------------------------------
    {
      if (slice_effect.exists())
        record_completion_effect(slice_effect);
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        completed_points += points;
        legion_assert(completed_points <= total_points);
        need_trigger = (completed_points == total_points);
        if (metadata != nullptr)
        {
          legion_assert(redop > 0);
          if (reduction_metadata == nullptr)
          {
            reduction_metadata = metadata;
            reduction_metasize = metasize;
            metadata = nullptr;  // mark that we grabbed it
          }
        }
      }
      if (need_trigger)
      {
        // If we are reducing to a single value we need to finish that now
        if ((redop > 0) && (predication_state != PREDICATED_FALSE_STATE))
        {
          legion_assert(
              (serdez_redop_fns != nullptr) || !reduction_instances.empty());
          legion_assert(
              (serdez_redop_fns != nullptr) ||
              (reduction_instance == reduction_instances.front()));
          // First finish applying any deterministic reductions
          if (deterministic_redop)
          {
            // Fold any temporary future for deterministic reduction
            for (std::map<DomainPoint, std::pair<FutureInstance*, ApEvent>>::
                     iterator it = temporary_futures.begin();
                 it != temporary_futures.end();
                 /*nothing*/)
            {
              if (fold_reduction_future(it->second.first, it->second.second))
              {
                delete it->second.first;
                std::map<DomainPoint, std::pair<FutureInstance*, ApEvent>>::
                    iterator to_delete = it++;
                temporary_futures.erase(to_delete);
              }
              else
                it++;
            }
          }
          else if (serdez_redop_fns == nullptr)
          {
            // Merge any reduction fold events back into the
            // reduction_instance_precondition to know when the
            // reduction instance is safe to use
            // Note all these events dominate the reduction fold precondition
            // so there is no need to include and we can just overwrite it
            if (!reduction_fold_effects.empty())
            {
              reduction_instance_precondition =
                  Runtime::merge_events(nullptr, reduction_fold_effects);
              reduction_fold_effects.clear();
            }
          }
          legion_assert(reduction_fold_effects.empty());
          // Finish the index task reduction
          finish_index_task_reduction();
        }
        // Forward completion effects for any local-mapped slices
        for (SliceTask* const slice : origin_mapped_slices)
          slice->forward_completion_effects();
        complete_execution();
      }
      // If we didn't grab ownership then free this now
      if (metadata != nullptr)
        free(metadata);
    }

    //--------------------------------------------------------------------------
    void IndexTask::finish_index_task_reduction(void)
    //--------------------------------------------------------------------------
    {
      // If we have serdez redop fns, we now know how big the output
      // is so we can make our target instances and complete the mapping
      if (serdez_redop_fns != nullptr)
      {
        legion_assert(reduction_instances.empty());
        legion_assert(!serdez_redop_targets.empty());
        reduction_instances.reserve(serdez_redop_targets.size());
        int runtime_visible_index = -1;
        TaskTreeCoordinates coordinates;
        for (const Memory& memory : serdez_redop_targets)
        {
          if ((runtime_visible_index == -1) &&
              (memory == runtime->runtime_system_memory))
          {
            runtime_visible_index = reduction_instances.size();
            reduction_instances.emplace_back(FutureInstance::create_local(
                serdez_redop_state, serdez_redop_state_size, false /*own*/));
          }
          else
          {
            if (coordinates.empty())
              compute_task_tree_coordinates(coordinates);
            MemoryManager* manager = runtime->find_memory_manager(memory);
            // Safe to block here indefinitely waiting for unbounded pools
            reduction_instances.emplace_back(manager->create_future_instance(
                unique_op_id, coordinates, serdez_redop_state_size,
                nullptr /*safe_for_unbounded_pools*/));
          }
        }
        if (runtime_visible_index < 0)
        {
          runtime_visible_index = reduction_instances.size();
          reduction_instances.emplace_back(FutureInstance::create_local(
              serdez_redop_state, serdez_redop_state_size, false /*own*/));
        }
        // Make sure the instance with the data is at the front
        if (runtime_visible_index > 0)
          std::swap(
              reduction_instances.front(),
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
      legion_assert(!reduction_instances.empty());
      legion_assert(reduction_instance == reduction_instances.front());
      legion_assert(reduction_fold_effects.empty());
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
          if (reduction_instances.size() <= (2 * idx))
            break;
          reduction_instances_ready[2 * idx] =
              reduction_instances[2 * idx]->copy_from(
                  reduction_instances[idx], this,
                  reduction_instances_ready[idx]);
          if (reduction_instances.size() <= (2 * idx + 1))
            break;
          reduction_instances_ready[2 * idx + 1] =
              reduction_instances[2 * idx + 1]->copy_from(
                  reduction_instances[idx], this,
                  reduction_instances_ready[idx]);
        }
        record_completion_effects(reduction_instances_ready);
      }
      else
        record_completion_effect(reduction_instance_precondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::return_slice_commit(
        unsigned points, RtEvent commit_precondition)
    //--------------------------------------------------------------------------
    {
      bool need_trigger = false;
      {
        AutoLock o_lock(op_lock);
        if (commit_precondition.exists())
          commit_preconditions.insert(commit_precondition);
        committed_points += points;
        legion_assert(committed_points <= total_points);
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
    void IndexTask::unpack_point_mapped(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      DomainPoint point;
      derez.deserialize(point);
      RtEvent mapped_event;
      derez.deserialize(mapped_event);
      return_point_mapped(point, mapped_event);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_complete(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      ApEvent completion_effect;
      derez.deserialize(completion_effect);
      if (redop > 0)
      {
        legion_assert(reduction_op != nullptr);
        if (deterministic_redop)
        {
          size_t num_futures;
          derez.deserialize(num_futures);
          for (unsigned idx = 0; idx < num_futures; idx++)
          {
            DomainPoint point;
            derez.deserialize(point);
            FutureInstance* instance = FutureInstance::unpack_instance(derez);
            ApEvent effects;
            if (!instance->is_meta_visible)
              derez.deserialize(effects);
            reduce_future(point, instance, effects);
          }
        }
        else
        {
          if (serdez_redop_fns != nullptr)
          {
            size_t reduc_size;
            derez.deserialize(reduc_size);
            if (reduc_size > 0)
            {
              const void* reduc_ptr = derez.get_current_pointer();
              FutureInstance instance(
                  reduc_ptr, reduc_size, true /*external*/,
                  false /*own allocation*/);
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
              FutureInstance* instance = FutureInstance::unpack_instance(derez);
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
          if (reduction_metadata == nullptr)
          {
            reduction_metadata = malloc(metasize);
            memcpy(reduction_metadata, derez.get_current_pointer(), metasize);
            reduction_metasize = metasize;
          }
          derez.advance_pointer(metasize);
        }
      }
      return_slice_complete(points, completion_effect);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_commit(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      size_t points;
      derez.deserialize(points);
      RtEvent commit_precondition;
      derez.deserialize(commit_precondition);
      const RtEvent resources_returned =
          (must_epoch == nullptr) ?
              ResourceTracker::unpack_resources_return(derez, parent_ctx) :
              ResourceTracker::unpack_resources_return(derez, must_epoch);
      if (resources_returned.exists())
      {
        if (commit_precondition.exists())
          return_slice_commit(
              points,
              Runtime::merge_events(commit_precondition, resources_returned));
        else
          return_slice_commit(points, resources_returned);
      }
      else
        return_slice_commit(points, commit_precondition);
    }

    //--------------------------------------------------------------------------
    void IndexTask::unpack_slice_collective_versioning_rendezvous(
        Deserializer& derez, unsigned index, size_t total_points)
    //--------------------------------------------------------------------------
    {
      bool done = false;
      op::map<LogicalRegion, RegionVersioning> to_perform;
      {
        size_t num_regions;
        derez.deserialize(num_regions);
        AutoLock o_lock(op_lock);
        std::map<unsigned, PendingVersioning>::iterator finder =
            pending_versioning.find(index);
        if (finder == pending_versioning.end())
        {
          finder = pending_versioning
                       .insert(std::make_pair(index, PendingVersioning()))
                       .first;
          finder->second.remaining_arrivals = this->get_collective_points();
        }
        for (unsigned idx1 = 0; idx1 < num_regions; idx1++)
        {
          LogicalRegion region;
          derez.deserialize(region);
          RtUserEvent ready_event;
          derez.deserialize(ready_event);
          op::map<LogicalRegion, RegionVersioning>::iterator region_finder =
              finder->second.region_versioning.find(region);
          if (region_finder == finder->second.region_versioning.end())
          {
            region_finder =
                finder->second.region_versioning
                    .emplace(std::make_pair(region, RegionVersioning()))
                    .first;
            region_finder->second.ready_event = ready_event;
          }
          else
            Runtime::trigger_event(
                ready_event, region_finder->second.ready_event);
          size_t num_trackers;
          derez.deserialize(num_trackers);
          for (unsigned idx2 = 0; idx2 < num_trackers; idx2++)
          {
            std::pair<AddressSpaceID, EqSetTracker*> key;
            derez.deserialize(key.first);
            derez.deserialize(key.second);
            legion_assert(
                region_finder->second.trackers.find(key) ==
                region_finder->second.trackers.end());
            derez.deserialize(region_finder->second.trackers[key]);
          }
        }
        legion_assert(finder->second.remaining_arrivals >= total_points);
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
        finalize_collective_versioning_analysis(
            index, parent_req_index, to_perform);
      }
    }

    //--------------------------------------------------------------------------
    void IndexTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(is_replaying());
      legion_assert(current_proc.exists());
      LegionSpy::log_replay_operation(unique_op_id);
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
        runtime->find_domain(internal_space, internal_domain);
        enumerate_futures(internal_domain);
      }
      if (concurrent_task)
        tpl->initialize_concurrent_groups(this);
      // Mark that this is origin mapped effectively in case we
      // have any remote tasks, do this before we clone it
      map_origin = true;
      SliceTask* new_slice =
          this->clone_as_slice_task(internal_space, current_proc, false, false);
      // Count how many total points we need for this index space task
      total_points = new_slice->enumerate_points(false /*inline*/);
      // We need to make one slice per point here in case we need to move
      // points to remote nodes. The way we do slicing right now prevents
      // us from knowing which point tasks are going remote until later in
      // the replay so we have to be pessimistic here
      new_slice->expand_replay_slices(slices);
      // Then do the replay on all the slices
      for (SliceTask* const slice : slices) slice->trigger_replay();
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceRemoteMapped::handle(
        Deserializer& derez, AddressSpaceID source)
    //--------------------------------------------------------------------------
    {
      IndexTask* task;
      derez.deserialize(task);
      task->unpack_point_mapped(derez, source);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceRemoteComplete::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      IndexTask* task;
      derez.deserialize(task);
      task->unpack_slice_complete(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceRemoteCommit::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      IndexTask* task;
      derez.deserialize(task);
      task->unpack_slice_commit(derez);
    }

    //--------------------------------------------------------------------------
    /*static*/ void SliceFindIntraDependence::handle(
        Deserializer& derez, AddressSpaceID)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      IndexTask* task;
      derez.deserialize(task);
      DomainPoint point;
      derez.deserialize(point);
      RtUserEvent to_trigger;
      derez.deserialize(to_trigger);
      task->find_pointwise_dependence(
          point, task->get_generation(), true /*intra space*/, to_trigger);
    }

    //--------------------------------------------------------------------------
    void IndexTask::start_check_point_requirements(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        const RegionRequirement& req = regions[idx];
        if ((!IS_WRITE(req) && !IS_OUTPUT_DISCARD(req)) || IS_COLLECTIVE(req))
          continue;
        // If the projection functions are invertible then we don't have to
        // worry about interference because the runtime knows how to hook
        // up those kinds of dependences
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
        {
          if (req.projection == 0)
          {
            if (req.handle_type == LEGION_PARTITION_PROJECTION)
            {
              IndexPartNode* partition =
                  runtime->get_node(req.partition.get_index_partition());
              if (partition->is_disjoint())
                continue;
            }
            else  // Identity functor is invertible
              continue;
          }
          else
          {
            ProjectionFunction* func =
                runtime->find_projection_function(req.projection);
            if (func->is_invertible)
              continue;
          }
        }
        interfering_requirements.insert(
            std::pair<unsigned, unsigned>(idx, idx));
      }
      if (interfering_requirements.empty() || launch_space->is_empty())
        return;
      Domain internal_domain;
      if (internal_space.exists())
        runtime->find_domain(internal_space, internal_domain);
      Domain launch_domain = launch_space->get_tight_domain();
      // Exchange all the domains for the interfering requirements
      std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>
          point_domains;
      for (const std::pair<unsigned, unsigned>& req_pair :
           interfering_requirements)
      {
        std::vector<std::pair<DomainPoint, Domain>>& domains =
            point_domains[req_pair.first];
        // Already found it for this region requirements
        if (!domains.empty())
          continue;
        domains.reserve(internal_domain.get_volume());
        const RegionRequirement& req = regions[req_pair.first];
        ProjectionFunction* function = nullptr;
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
          function = runtime->find_projection_function(req.projection);
        for (Domain::DomainPointIterator itr(internal_domain); itr; itr++)
        {
          LogicalRegion point_region =
              (function == nullptr) ?
                  req.region :
                  function->project_point(
                      this, req_pair.first, launch_domain, *itr);
          if (!point_region.exists())
            continue;
          Domain point_domain;
          runtime->find_domain(point_region.get_index_space(), point_domain);
          domains.emplace_back(std::make_pair(*itr, point_domain));
        }
      }
      finish_check_point_requirements(point_domains);
    }

    //--------------------------------------------------------------------------
    void IndexTask::finish_check_point_requirements(
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>&
            point_domains)
    //--------------------------------------------------------------------------
    {
      Domain internal_domain;
      if (internal_space.exists())
        runtime->find_domain(internal_space, internal_domain);
      Domain launch_domain = launch_space->get_tight_domain();
      // Iterate our local points and check their first region requirements
      // against all the points in the second region requirements
      for (const std::pair<unsigned, unsigned>& req_pair :
           interfering_requirements)
      {
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>::
            const_iterator finder = point_domains.find(req_pair.first);
        legion_assert(finder != point_domains.end());
        const RegionRequirement& req = get_requirement(req_pair.second);
        ProjectionFunction* function = nullptr;
        if (req.handle_type != LEGION_SINGULAR_PROJECTION)
          function = runtime->find_projection_function(req.projection);
        for (Domain::DomainPointIterator itr(internal_domain); itr; itr++)
        {
          LogicalRegion point_region =
              (function == nullptr) ?
                  req.region :
                  function->project_point(
                      this, req_pair.second, launch_domain, *itr);
          if (!point_region.exists())
            continue;
          IndexSpaceNode* node =
              runtime->get_node(point_region.get_index_space());
          DomainPoint interfering;
          if (node->has_interfering_point(
                  finder->second, interfering,
                  (req_pair.first == req_pair.second) ? *itr : DomainPoint()))
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Index " << *this
                  << " has interfering region requirements between "
                  << "region requirements " << req_pair.first
                  << " of point task " << interfering << " and "
                  << req_pair.second << " of point task " << *itr
                  << ". Interfering region requirements are not permitted for "
                     "index tasks.";
            error.raise();
          }
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // OutputExtentExchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    OutputExtentExchange::OutputExtentExchange(
        ReplicateContext* ctx, ReplIndexTask* own, CollectiveIndexLocation loc,
        std::vector<OutputRegionState>& all_states)
      : AllGatherCollective<false>(loc, ctx), owner(own),
        all_output_states(all_states)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    OutputExtentExchange::~OutputExtentExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void OutputExtentExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      rez.serialize(all_output_states.size());
#endif
      for (const OutputRegionState& state : all_output_states)
      {
        rez.serialize(state.points.size());
        for (const std::pair<const DomainPoint, DomainPoint>& it : state.points)
        {
          rez.serialize(it.first);
          rez.serialize(it.second);
        }
      }
    }

    //--------------------------------------------------------------------------
    void OutputExtentExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_DEBUG
      size_t num_sizes;
      derez.deserialize(num_sizes);
#endif
      legion_assert(all_output_states.size() == num_sizes);
      for (unsigned idx = 0; idx < all_output_states.size(); idx++)
      {
        OutputRegionState& state = all_output_states[idx];
        size_t num_entries;
        derez.deserialize(num_entries);
        for (unsigned eidx = 0; eidx < num_entries; eidx++)
        {
          DomainPoint point;
          derez.deserialize(point);
          derez.deserialize(state.points[point]);
        }
      }
    }

    //--------------------------------------------------------------------------
    RtEvent OutputExtentExchange::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      owner->finalize_output_regions(false /*first invocation*/);
      return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Concurrent Mapping Rendezvous
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ConcurrentMappingRendezvous::ConcurrentMappingRendezvous(
        ReplIndexTask* own, CollectiveIndexLocation loc, ReplicateContext* ctx,
        std::map<Color, MultiTask::ConcurrentGroup>& g)
      : AllGatherCollective<true>(loc, ctx), owner(own), groups(g)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ConcurrentMappingRendezvous::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(groups.size());
      for (std::pair<const Color, MultiTask::ConcurrentGroup>& group : groups)
      {
        rez.serialize(group.first);
        if (group.second.preconditions.empty())
          rez.serialize(RtEvent::NO_RT_EVENT);
        else if (group.second.preconditions.size() == 1)
          rez.serialize(group.second.preconditions.front());
        else
        {
          RtEvent pre = Runtime::merge_events(group.second.preconditions);
          rez.serialize(pre);
          group.second.preconditions.resize(1);
          group.second.preconditions[0] = pre;
        }
        rez.serialize<size_t>(group.second.shards.size());
        for (const ShardID& shard : group.second.shards) rez.serialize(shard);
        rez.serialize<size_t>(group.second.processors.size());
        for (const std::pair<const Processor, DomainPoint>& proc_pair :
             group.second.processors)
        {
          rez.serialize(proc_pair.first);
          rez.serialize(proc_pair.second);
        }
        rez.serialize(group.second.color_points);
      }
      rez.serialize<size_t>(trace_barriers.size());
      for (const std::pair<const Color, std::pair<RtBarrier, size_t>>&
               barrier_pair : trace_barriers)
      {
        rez.serialize(barrier_pair.first);
        rez.serialize(barrier_pair.second.first);
        rez.serialize(barrier_pair.second.second);
      }
    }

    //--------------------------------------------------------------------------
    void ConcurrentMappingRendezvous::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_groups;
      derez.deserialize(num_groups);
      for (unsigned idx1 = 0; idx1 < num_groups; idx1++)
      {
        Color color;
        derez.deserialize(color);
        MultiTask::ConcurrentGroup& group = groups[color];
        RtEvent precondition;
        derez.deserialize(precondition);
        if (precondition.exists())
          group.preconditions.emplace_back(precondition);
        size_t num_shards;
        derez.deserialize(num_shards);
        for (unsigned idx2 = 0; idx2 < num_shards; idx2++)
        {
          ShardID shard;
          derez.deserialize(shard);
          if (!std::binary_search(
                  group.shards.begin(), group.shards.end(), shard))
          {
            group.shards.emplace_back(shard);
            std::sort(group.shards.begin(), group.shards.end());
          }
        }
        size_t num_processors;
        derez.deserialize(num_processors);
        for (unsigned idx2 = 0; idx2 < num_processors; idx2++)
        {
          Processor proc;
          derez.deserialize(proc);
          DomainPoint point;
          derez.deserialize(point);
          std::map<Processor, DomainPoint>::const_iterator finder =
              group.processors.find(proc);
          if (finder != group.processors.end())
          {
            // If we are a non-participating shard we might get our
            // own points sent back to us so we need to detect that
            // case and not report an error for it
            legion_assert(
                (finder->second != point) || (!participating && (stage == -1)));
            if (finder->second != point)
              owner->report_concurrent_mapping_failure(
                  proc, point, finder->second);
          }
          else
            group.processors.emplace(std::make_pair(proc, point));
        }
        size_t points;
        derez.deserialize(points);
        if (!participating)
        {
          legion_assert(stage == -1);
          group.color_points = points;
        }
        else
          group.color_points += points;
      }
      size_t num_barriers;
      derez.deserialize(num_barriers);
      for (unsigned idx = 0; idx < num_barriers; idx++)
      {
        Color color;
        derez.deserialize(color);
        legion_assert(trace_barriers.find(color) == trace_barriers.end());
        std::pair<RtBarrier, size_t>& barrier = trace_barriers[color];
        derez.deserialize(barrier.first);
        derez.deserialize(barrier.second);
      }
    }

    //--------------------------------------------------------------------------
    void ConcurrentMappingRendezvous::set_trace_barrier(
        Color color, RtBarrier bar, size_t arrivals)
    //--------------------------------------------------------------------------
    {
      legion_assert(bar.exists());
      legion_assert(trace_barriers.find(color) == trace_barriers.end());
      trace_barriers.emplace(
          std::make_pair(color, std::make_pair(bar, arrivals)));
    }

    //--------------------------------------------------------------------------
    void ConcurrentMappingRendezvous::perform_rendezvous(void)
    //--------------------------------------------------------------------------
    {
      // Record our local shard as one of the shards for each group
      // In some cases we might already have computed it so we don't need
      // to add ourselves in that case as we'll already be represented
      for (std::pair<const Color, MultiTask::ConcurrentGroup>& group : groups)
      {
        legion_assert(group.second.group_points > 0);
        legion_assert(
            group.second.shards.empty() ||
            std::binary_search(
                group.second.shards.begin(), group.second.shards.end(),
                local_shard));
        group.second.color_points = group.second.group_points;
        if (group.second.shards.empty())
          group.second.shards.emplace_back(local_shard);
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    RtEvent ConcurrentMappingRendezvous::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      owner->finish_concurrent_mapped(trace_barriers);
      return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Concurrent Allreduce
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ConcurrentAllreduce::ConcurrentAllreduce(
        ReplicateContext* ctx, CollectiveID id, Color c,
        const std::vector<ShardID>& shards)
      : AllGatherCollective<false>(ctx, id, shards), color(c)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ConcurrentAllreduce::ConcurrentAllreduce(
        CollectiveIndexLocation loc, ReplicateContext* ctx)
      : AllGatherCollective<false>(loc, ctx), color(0)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ConcurrentAllreduce::~ConcurrentAllreduce(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ConcurrentAllreduce::perform_concurrent_allreduce(
        MultiTask::ConcurrentGroup& group)
    //--------------------------------------------------------------------------
    {
      slice_tasks.swap(group.slice_tasks);
      task_barrier = group.task_barrier;
      lamport_clock = group.lamport_clock;
      variant = group.variant;
      poisoned = group.poisoned;
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void ConcurrentAllreduce::perform_concurrent_allreduce(
        std::vector<std::pair<IndividualTask*, AddressSpaceID>>& single,
        std::vector<std::pair<SliceTask*, AddressSpace>>& slices,
        uint64_t clock, bool poison)
    //--------------------------------------------------------------------------
    {
      single_tasks.swap(single);
      slice_tasks.swap(slices);
      lamport_clock = clock;
      variant = 0;
      poisoned = poison;
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void ConcurrentAllreduce::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize(task_barrier);
      rez.serialize(lamport_clock);
      rez.serialize(variant);
      rez.serialize<bool>(poisoned);
    }

    //--------------------------------------------------------------------------
    void ConcurrentAllreduce::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      RtBarrier barrier;
      derez.deserialize(barrier);
      if (!task_barrier.exists() && barrier.exists())
        task_barrier = barrier;
      uint64_t clock;
      derez.deserialize(clock);
      if (lamport_clock < clock)
        lamport_clock = clock;
      VariantID vid;
      derez.deserialize(vid);
      if (variant != vid)
        variant = std::min(variant, vid);
      bool poison;
      derez.deserialize<bool>(poison);
      if (poison)
        poisoned = true;
    }

    //--------------------------------------------------------------------------
    RtEvent ConcurrentAllreduce::post_complete_exchange(void)
    //--------------------------------------------------------------------------
    {
      // Pull these onto the stack in order to avoid a race with the caller
      // cleaning up this object
      std::vector<std::pair<IndividualTask*, AddressSpaceID>> local_tasks;
      local_tasks.swap(single_tasks);
      std::vector<std::pair<SliceTask*, AddressSpace>> local_slices;
      local_slices.swap(slice_tasks);
      for (const std::pair<IndividualTask*, AddressSpaceID>& task_pair :
           local_tasks)
      {
        if (task_pair.second != runtime->address_space)
        {
          IndividualTaskConcurrentResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(task_pair.first);
            rez.serialize(lamport_clock);
            rez.serialize(poisoned);
          }
          rez.dispatch(task_pair.second);
        }
        else
          task_pair.first->finish_concurrent_allreduce(lamport_clock, poisoned);
      }
      for (const std::pair<SliceTask*, AddressSpaceID>& slice_pair :
           local_slices)
      {
        if (slice_pair.second != runtime->address_space)
        {
          SliceConcurrentResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(slice_pair.first);
            rez.serialize(color);
            rez.serialize(task_barrier);
            rez.serialize(lamport_clock);
            rez.serialize(variant);
            rez.serialize(poisoned);
          }
          rez.dispatch(slice_pair.second);
        }
        else
          slice_pair.first->finish_concurrent_allreduce(
              color, lamport_clock, poisoned, variant, task_barrier);
      }
      return RtEvent::NO_RT_EVENT;
    }

    /////////////////////////////////////////////////////////////
    // Repl Index Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplIndexTask::ReplIndexTask(void) : ReplCollectiveViewCreator<IndexTask>()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplIndexTask::~ReplIndexTask(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplIndexTask::activate(void)
    //--------------------------------------------------------------------------
    {
      ReplCollectiveViewCreator<IndexTask>::activate();
      sharding_functor = std::numeric_limits<unsigned>::max();
      sharding_function = nullptr;
      serdez_redop_collective = nullptr;
      all_reduce_collective = nullptr;
      reduction_collective = nullptr;
      broadcast_collective = nullptr;
      output_size_collective = nullptr;
      collective_check_id = 0;
      interfering_check_id = 0;
      slice_sharding_output = false;
      output_bar = RtBarrier::NO_RT_BARRIER;
      concurrent_mapping_rendezvous = nullptr;
      interfering_check_id = 0;
      interfering_exchange = nullptr;
      collective_exchange_id = 0;
      collective_exchange = nullptr;
      sharding_collective = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      for (std::pair<const Color, ConcurrentGroup>& group : concurrent_groups)
        if (group.second.exchange == nullptr)
          delete group.second.exchange;
      if (serdez_redop_collective != nullptr)
        delete serdez_redop_collective;
      if (all_reduce_collective != nullptr)
        delete all_reduce_collective;
      if (reduction_collective != nullptr)
        delete reduction_collective;
      if (broadcast_collective != nullptr)
        delete broadcast_collective;
      if (output_size_collective != nullptr)
        delete output_size_collective;
      if (concurrent_mapping_rendezvous != nullptr)
        delete concurrent_mapping_rendezvous;
      if (interfering_exchange != nullptr)
        delete interfering_exchange;
      concurrent_exchange_ids.clear();
      if (collective_exchange != nullptr)
        delete collective_exchange;
      if (sharding_collective != nullptr)
        delete sharding_collective;
      unique_intra_space_deps.clear();
      ReplCollectiveViewCreator<IndexTask>::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::prepare_map_must_epoch(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      legion_assert(must_epoch != nullptr);
      legion_assert(sharding_function != nullptr);
      set_origin_mapped(true);
      total_points = launch_space->get_volume();
      if (!elide_future_return)
      {
        future_map = must_epoch->get_future_map();
        const IndexSpace local_space =
            sharding_space.exists() ?
                sharding_function->find_shard_space(
                    repl_ctx->owner_shard->shard_id, launch_space,
                    sharding_space, get_provenance()) :
                sharding_function->find_shard_space(
                    repl_ctx->owner_shard->shard_id, launch_space,
                    launch_space->handle, get_provenance());
        if (local_space.exists())
        {
          Domain local_domain;
          runtime->find_domain(local_space, local_domain);
          enumerate_futures(local_domain);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);

      // We might be able to skip this if the sharding function was already
      // picked for us which occurs when we're part of a must-epoch launch
      if (sharding_function == nullptr)
        select_sharding_function(repl_ctx);
      legion_assert(sharding_function != nullptr);
      if (runtime->safe_mapper)
      {
        legion_assert(sharding_collective != nullptr);
        sharding_collective->contribute(this->sharding_functor);
        if (sharding_collective->is_target() &&
            !sharding_collective->validate(this->sharding_functor))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << mapper->get_mapper_name()
                << " chose different sharding functions for index task "
                << get_task_name() << " (UID " << get_unique_id() << ") in "
                << parent_ctx->get_task_name() << " (UID "
                << parent_ctx->get_unique_id() << ").";
          error.raise();
        }
      }
      // Now we can do the normal prepipeline stage
      IndexTask::trigger_prepipeline_stage();
      if (runtime->safe_mapper)
      {
        // Check that all the mappers agreed on the set of
        // collective view region requirements
        if (repl_ctx->owner_shard->shard_id == 0)
        {
          Serializer rez;
          rez.serialize<size_t>(check_collective_regions.size());
          for (const unsigned& region_idx : check_collective_regions)
            rez.serialize(region_idx);
          BufferBroadcast collective(collective_check_id, repl_ctx);
          collective.broadcast(
              const_cast<void*>(rez.get_buffer()), rez.get_used_bytes(),
              false /*copy*/);
        }
        else
        {
          BufferBroadcast collective(
              collective_check_id, 0 /*owner*/, repl_ctx);
          size_t size;
          const void* buffer = collective.get_buffer(size);
          Deserializer derez(buffer, size);
          size_t num_regions;
          derez.deserialize(num_regions);
          if (num_regions != check_collective_regions.size())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Mapper " << *mapper << " provided different number of "
                << "logical regions to check for collective views on shards 0 "
                << "and " << repl_ctx->owner_shard->shard_id << " of task "
                << *this << ". Shard 0 provided " << num_regions
                << " regions while Shard " << repl_ctx->owner_shard->shard_id
                << " provided " << check_collective_regions.size()
                << " regions. All shards must provide the same logical "
                << "regions to check for the collective view creation.";
            error.raise();
          }
          for (unsigned idx = 0; idx < num_regions; idx++)
          {
            unsigned index;
            derez.deserialize(index);
            if (!std::binary_search(
                    check_collective_regions.begin(),
                    check_collective_regions.end(), index))
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error << "Mapper " << *mapper << " provided different logical "
                    << "regions to check for collective views on shards 0 and "
                    << repl_ctx->owner_shard->shard_id << " of task " << *this
                    << ". Shard 0 provided region " << index << " while Shard "
                    << repl_ctx->owner_shard->shard_id
                    << " did not. All shards "
                    << "must provide the same logical regions to check for the "
                    << "collective view creation.";
              error.raise();
            }
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::select_sharding_function(ReplicateContext* repl_ctx)
    //--------------------------------------------------------------------------
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
              << " failed to pick a valid sharding functor for task "
              << get_task_name() << " (UID " << get_unique_id() << ").";
        error.raise();
      }
      this->sharding_functor = output.chosen_functor;
      sharding_function =
          repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      slice_sharding_output = output.slice_recurse;
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finish_check_point_requirements(
        std::map<unsigned, std::vector<std::pair<DomainPoint, Domain>>>&
            point_domains)
    //--------------------------------------------------------------------------
    {
      // See if this is the first time through or not
      if (interfering_exchange == nullptr)
      {
        // First time through, make the exchange and kick it off
        legion_assert(interfering_check_id > 0);
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        interfering_exchange = new InterferingPointExchange<ReplIndexTask>(
            repl_ctx, interfering_check_id, this);
        // Record a dependence on this to make sure it is done before we
        // clean up the operation
        commit_preconditions.insert(interfering_exchange->get_done_event());
        interfering_exchange->exchange_domain_points(point_domains);
      }
      else  // Second time through call the base class since we have the
            // results
        IndexTask::finish_check_point_requirements(point_domains);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // If we have a future map then set the sharding function
      if ((redop == 0) && !elide_future_return && (must_epoch == nullptr))
      {
        legion_assert(future_map.impl != nullptr);
        ReplFutureMapImpl* impl =
            legion_safe_cast<ReplFutureMapImpl*>(future_map.impl);
        impl->set_sharding_function(sharding_function);
      }
      // Compute the local index space of points for this shard
      if (sharding_space.exists())
        internal_space = sharding_function->find_shard_space(
            repl_ctx->owner_shard->shard_id, launch_space, sharding_space,
            get_provenance());
      else
        internal_space = sharding_function->find_shard_space(
            repl_ctx->owner_shard->shard_id, launch_space, launch_space->handle,
            get_provenance());
      if (runtime->safe_model)
        start_check_point_requirements();
      // If we're recording then record the local_space
      if (is_recording())
      {
        legion_assert(!is_remote());
        legion_assert((tpl != nullptr) && tpl->is_recording());
        tpl->record_local_space(trace_local_id, internal_space);
        // Record the sharding function if needed for the future map
        if (redop == 0)
          tpl->record_sharding_function(trace_local_id, sharding_function);
      }
      // If it's empty we're done, otherwise we go back on the queue
      if (!internal_space.exists())
      {
        // Check to see if we still need to participate in the premap_task call
        if (must_epoch == nullptr)
          premap_task();
        // Still need to participate in any collective mappings
        if (concurrent_task || !check_collective_regions.empty())
        {
          collective_exchange =
              new AllReduceCollective<MaxReduction<uint64_t>, false>(
                  repl_ctx, collective_exchange_id);
          collective_exchange->async_all_reduce(collective_lamport_clock);
          AutoLock o_lock(op_lock);
          commit_preconditions.insert(collective_exchange->get_done_event());
        }
        // Still need to participate in any collective view rendezvous
        if (!collective_view_rendezvous.empty())
          shard_off_collective_rendezvous(commit_preconditions);
        // Still have to do this for legion spy
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        // Finalize any output regions
        if (output_size_collective != nullptr)
        {
          finalize_output_regions(true /*first invocation*/);
          record_output_registered(RtEvent::NO_RT_EVENT);
        }
        // We have no local points, so we can just trigger
        if (serdez_redop_fns == nullptr)
        {
          if (!map_applied_conditions.empty())
            complete_mapping(Runtime::merge_events(map_applied_conditions));
          else
            complete_mapping();
        }
        if (concurrent_task)
          concurrent_mapping_rendezvous->perform_rendezvous();
        if (redop > 0)
          finish_index_task_reduction();
        complete_execution();
        trigger_children_committed();
      }
      else  // We have valid points, so it goes on the ready queue
      {
        // If we had a trace then we might need to recompute our parent
        // region indexes now since we might have skipped it earlier
        if (trace != nullptr)
          compute_parent_indexes(true /*force*/);
        // Update the total number of points we're actually repsonsible
        // for now with this shard
        IndexSpaceNode* node = runtime->get_node(internal_space);
        total_points = node->get_volume();
        legion_assert(total_points > 0);
        if ((redop == 0) && !elide_future_return)
        {
          Domain shard_domain = node->get_tight_domain();
          enumerate_futures(shard_domain);
        }
        if (concurrent_task)
        {
          if (concurrent_functor == 0)
          {
            // The built-in concurrent coloring functor makes this easy
            // since it always maps all the points to the same color
            ConcurrentGroup& group = concurrent_groups[0];
            group.precondition.interpreted = Runtime::create_rt_user_event();
            group.color_points = launch_space->get_volume();
          }
          else
          {
            // Not the built-in functor so we actually need to some work
            ConcurrentColoringFunctor* functor =
                runtime->find_concurrent_coloring_functor(concurrent_functor);
            Domain shard_domain = node->get_tight_domain();
            for (Domain::DomainPointIterator itr(shard_domain); itr; itr++)
            {
              Color color = functor->color(*itr, index_domain);
              if (concurrent_groups.find(color) == concurrent_groups.end())
                concurrent_groups[color].precondition.interpreted =
                    Runtime::create_rt_user_event();
            }
          }
        }
        // If we still need to slice the task then we can run it
        // through the normal path, otherwise we can simply make
        // the slice task for these points and put it in the queue
        if (!slice_sharding_output)
        {
          if (must_epoch == nullptr)
            premap_task();
          SliceTask* new_slice = this->clone_as_slice_task(
              internal_space, target_proc, false /*recurse*/,
              !runtime->stealing_disabled);
          slices.emplace_back(new_slice);
          trigger_slices();
        }
        else
          enqueue_ready_operation();
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_replay(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(tpl != nullptr);
      elide_collective_rendezvous();
      internal_space = tpl->find_local_space(trace_local_id);
      if ((redop == 0) && !elide_future_return)
      {
        sharding_function = tpl->find_sharding_function(trace_local_id);
        legion_assert(future_map.impl != nullptr);
        ReplFutureMapImpl* impl =
            legion_safe_cast<ReplFutureMapImpl*>(future_map.impl);
        impl->set_sharding_function(sharding_function);
      }
      // We know all the points are going to be issues so no need for this
      if (concurrent_mapping_rendezvous != nullptr)
        concurrent_mapping_rendezvous->elide_collective();
      // If it's empty we're done, otherwise we do the replay
      if (!internal_space.exists())
      {
        LegionSpy::log_replay_operation(unique_op_id);
        LegionSpy::log_operation_events(
            unique_op_id, ApEvent::NO_AP_EVENT, ApEvent::NO_AP_EVENT);
        // We have no local points, so we can just trigger
        if (serdez_redop_fns == nullptr)
        {
          if (!map_applied_conditions.empty())
            complete_mapping(Runtime::merge_events(map_applied_conditions));
          else
            complete_mapping();
        }
        if (redop > 0)
        {
          std::vector<Memory> reduction_futures;
          tpl->get_premap_output(this, reduction_futures);
          create_future_instances(reduction_futures);
          finish_index_task_reduction();
        }
        complete_execution();
        trigger_children_committed();
      }
      else
        IndexTask::trigger_replay();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      perform_base_dependence_analysis();
      ShardingFunction* analysis_sharding_function = sharding_function;
      if (must_epoch_task)
      {
        // Note we use a special
        // projection function for must epoch launches that maps all the
        // tasks to the special shard UINT_MAX so that they appear to be
        // on a different shard than any other tasks, but on the same shard
        // for all the tasks in the must epoch launch.
        analysis_sharding_function =
            repl_ctx->get_universal_sharding_function();
      }
      analyze_region_requirements(
          launch_space, analysis_sharding_function, sharding_space);
      if ((concurrent_task && !must_epoch_task) ||
          !check_collective_regions.empty())
      {
        // Create the collective exchange ID in case we need it
        collective_exchange_id = repl_ctx->get_next_collective_index(
            COLLECTIVE_LOC_68, true /*logical*/);
        // Generate any collective view rendezvous that we will need
        for (const unsigned& region_idx : check_collective_regions)
          create_collective_rendezvous(
              logical_regions[region_idx].parent.get_tree_id(), region_idx);
      }
      if (concurrent_task && !must_epoch_task)
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        // If this is a concurrent task we need to allocated collective IDs
        // for all colors that the task might need for when it goes to
        // perform its max allreduce. We need to allocate IDs for all colors
        // even if we might only use some of them so that we guarantee that
        // all the shards stay with the same set of collective IDs
        // If the concurrent coloring functor is the built-in one then we
        // know that we only need one ID in that case
        if (concurrent_functor > 0)
        {
          ConcurrentColoringFunctor* functor =
              runtime->find_concurrent_coloring_functor(concurrent_functor);
          if (functor->supports_max_color())
          {
            Color max_color = functor->max_color(index_domain);
            for (Color color = 0; color <= max_color; color++)
              concurrent_exchange_ids[color] =
                  repl_ctx->get_next_collective_index(
                      COLLECTIVE_LOC_79, true /*logical*/);
          }
          else
          {
            // If we have a trace we can try to look these up
            bool found = false;
            if (trace != nullptr)
              found =
                  trace->find_concurrent_colors(this, concurrent_exchange_ids);
            if (!found)
            {
              // Iterate all the points and save the unique colors
              for (Domain::DomainPointIterator itr(index_domain); itr; itr++)
              {
                Color color = functor->color(*itr, index_domain);
                if (concurrent_exchange_ids.find(color) ==
                    concurrent_exchange_ids.end())
                  concurrent_exchange_ids.emplace(
                      std::pair<Color, CollectiveID>(color, 0));
              }
              legion_assert(!concurrent_exchange_ids.empty());
              if (trace != nullptr)
                trace->record_concurrent_colors(this, concurrent_exchange_ids);
            }
            for (std::pair<const Color, CollectiveID>& id_pair :
                 concurrent_exchange_ids)
            {
              legion_assert(id_pair.second == 0);
              id_pair.second = repl_ctx->get_next_collective_index(
                  COLLECTIVE_LOC_79, true /*logical*/);
            }
          }
        }
        else
          concurrent_exchange_ids[0] = repl_ctx->get_next_collective_index(
              COLLECTIVE_LOC_79, true /*logical*/);
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::create_future_instances(
        std::vector<Memory>& target_memories)
    //--------------------------------------------------------------------------
    {
      // Do the base call first
      IndexTask::create_future_instances(target_memories);
      // Now check to see if we need to make a shadow instance for our
      // future all reduce collective
      if (all_reduce_collective != nullptr)
      {
        legion_assert(!reduction_instances.empty());
        legion_assert(reduction_instance != nullptr);
        // If the instance is in a memory we cannot see or is "too big"
        // then we need to make the shadow instance for the future
        // all-reduce collective to use now while still in the mapping stage
        if ((!reduction_instance.load()->is_meta_visible) ||
            (reduction_instance.load()->size > LEGION_MAX_RETURN_SIZE))
        {
          MemoryManager* manager =
              runtime->find_memory_manager(reduction_instance.load()->memory);
          TaskTreeCoordinates coordinates;
          compute_task_tree_coordinates(coordinates);
          // Safe to block indefinitely here waiting for unbounded pools
          FutureInstance* shadow_instance = manager->create_future_instance(
              unique_op_id, coordinates, reduction_op->sizeof_rhs,
              nullptr /*safe_for_unbounded_pools*/);
          all_reduce_collective->set_shadow_instance(shadow_instance);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finish_index_task_reduction(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(redop != 0);
      // Set the future if we actually ran the task or we speculated
      if (predication_state == PREDICATED_FALSE_STATE)
        return;
      if (serdez_redop_fns != nullptr)
      {
        legion_assert(serdez_redop_collective != nullptr);
        const std::map<ShardID, std::pair<void*, size_t>>& remote_buffers =
            serdez_redop_collective->exchange_buffers(
                serdez_redop_state, serdez_redop_state_size,
                deterministic_redop);
        if (deterministic_redop)
        {
          // Reset this back to empty so we can reduce in order across shards
          // Note the serdez_redop_collective took ownership of deleting
          // the buffer in this case so we know that it is not leaking
          serdez_redop_state = nullptr;
          for (const std::pair<const ShardID, std::pair<void*, size_t>>&
                   buffer_pair : remote_buffers)
          {
            if (serdez_redop_state == nullptr)
            {
              serdez_redop_state_size = buffer_pair.second.second;
              serdez_redop_state = malloc(serdez_redop_state_size);
              memcpy(
                  serdez_redop_state, buffer_pair.second.first,
                  serdez_redop_state_size);
            }
            else
              serdez_redop_fns->fold_fn(
                  reduction_op, serdez_redop_state, serdez_redop_state_size,
                  buffer_pair.second.first);
          }
        }
        else
        {
          for (const std::pair<const ShardID, std::pair<void*, size_t>>&
                   buffer_pair : remote_buffers)
          {
            legion_assert(
                buffer_pair.first != serdez_redop_collective->local_shard);
            serdez_redop_fns->fold_fn(
                reduction_op, serdez_redop_state, serdez_redop_state_size,
                buffer_pair.second.first);
          }
        }
      }
      else
      {
        legion_assert(
            (all_reduce_collective != nullptr) ||
            ((reduction_collective != nullptr) &&
             (broadcast_collective != nullptr)));
        legion_assert(!reduction_instances.empty());
        legion_assert(reduction_instance == reduction_instances.front());
        RtEvent collective_done;
        if (all_reduce_collective == nullptr)
        {
          reduction_collective->async_reduce(
              reduction_instance, reduction_instance_precondition);
          reduction_instance_precondition = broadcast_collective->finished;
          if (broadcast_collective->is_origin())
            collective_done = reduction_collective->get_done_event();
          else
            collective_done = broadcast_collective->async_broadcast(
                reduction_instance, ApEvent::NO_AP_EVENT,
                reduction_collective->get_done_event());
        }
        else
          collective_done = all_reduce_collective->async_reduce(
              reduction_instance, reduction_instance_precondition);
        // No need to do anything with the output local precondition
        // We already added it to the complete_effects when we made
        // the collective at the beginning
        if (collective_done.exists() && !collective_done.has_triggered())
        {
          AutoLock o_lock(op_lock);
          commit_preconditions.insert(collective_done);
        }
      }
      // Now call the base version of this to finish making
      // the instances for the future results
      IndexTask::finish_index_task_reduction();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      // Otherwise, we need to update the internal space so we only set
      // our local points with the predicate false result
      if (!elide_future_return)
      {
        if (redop == 0)
        {
          ReplicateContext* repl_ctx =
              legion_safe_cast<ReplicateContext*>(parent_ctx);
          legion_assert(sharding_function != nullptr);
          legion_assert(future_map.impl != nullptr);
          ReplFutureMapImpl* impl =
              legion_safe_cast<ReplFutureMapImpl*>(future_map.impl);
          impl->set_sharding_function(sharding_function);
          // Compute the local index space of points for this shard
          if (sharding_space.exists())
            internal_space = sharding_function->find_shard_space(
                repl_ctx->owner_shard->shard_id, launch_space, sharding_space,
                get_provenance());
          else
            internal_space = sharding_function->find_shard_space(
                repl_ctx->owner_shard->shard_id, launch_space,
                launch_space->handle, get_provenance());
        }
        else
        {
          if (serdez_redop_collective != nullptr)
            serdez_redop_collective->elide_collective();
          if (all_reduce_collective != nullptr)
            all_reduce_collective->elide_collective();
          if (reduction_collective != nullptr)
            reduction_collective->elide_collective();
          if (broadcast_collective != nullptr)
            broadcast_collective->elide_collective();
        }
      }
      if (output_size_collective != nullptr)
      {
        output_size_collective->elide_collective();
        runtime->phase_barrier_arrive(output_bar, 1 /*count*/);
      }
      elide_collective_rendezvous();
      // Now continue through and do the base case
      IndexTask::predicate_false();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      legion_assert(serdez_redop_collective == nullptr);
      legion_assert(all_reduce_collective == nullptr);
      legion_assert(reduction_collective == nullptr);
      legion_assert(broadcast_collective == nullptr);
      // If we have a reduction op then we need an exchange
      if (!elide_future_return && (redop > 0))
      {
        if (serdez_redop_fns == nullptr)
        {
          if (deterministic_redop)
          {
            broadcast_collective = new FutureBroadcastCollective(
                ctx, COLLECTIVE_LOC_63, 0 /*origin shard*/, this);
            reduction_collective = new FutureReductionCollective(
                ctx, COLLECTIVE_LOC_64, 0 /*origin shard*/, this,
                broadcast_collective, reduction_op, redop);
          }
          else
            all_reduce_collective = new FutureAllReduceCollective(
                this, COLLECTIVE_LOC_53, ctx, redop, reduction_op);
        }
        else
          serdez_redop_collective = new BufferExchange(ctx, COLLECTIVE_LOC_53);
      }
      if (!output_regions.empty())
      {
        bool has_unbound_region = false;
        bool has_local_indexing = false;
        for (unsigned idx = 0; idx < output_regions.size(); ++idx)
        {
          if (!output_region_options[idx].bounded_requirement())
          {
            has_unbound_region = true;
            if (!output_region_options[idx].global_indexing())
              has_local_indexing = true;
          }
        }
        if (has_unbound_region)
        {
          // We need an exchange if we've got a local index or we're
          // validating the programming model
          if (has_local_indexing || runtime->safe_model)
            output_size_collective = new OutputExtentExchange(
                ctx, this, COLLECTIVE_LOC_29, output_region_states);
          output_bar = ctx->get_next_output_regions_barrier();
        }
      }
      if (runtime->safe_mapper)
        collective_check_id = ctx->get_next_collective_index(COLLECTIVE_LOC_76);
      if (runtime->safe_model)
        interfering_check_id =
            ctx->get_next_collective_index(COLLECTIVE_LOC_69);
      if (concurrent_task && !must_epoch_task)
      {
        concurrent_mapping_rendezvous = new ConcurrentMappingRendezvous(
            this, COLLECTIVE_LOC_104, ctx, concurrent_groups);
        commit_preconditions.insert(
            concurrent_mapping_rendezvous->get_done_event());
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::set_sharding_function(
        ShardingID functor, ShardingFunction* function)
    //--------------------------------------------------------------------------
    {
      legion_assert(must_epoch != nullptr);
      legion_assert(sharding_function == nullptr);
      sharding_functor = functor;
      sharding_function = function;
    }

    //--------------------------------------------------------------------------
    FutureMap ReplIndexTask::create_future_map(
        TaskContext* ctx, IndexSpace launch_space, IndexSpace shard_space)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx = legion_safe_cast<ReplicateContext*>(ctx);
      IndexSpaceNode* launch_node = runtime->get_node(launch_space);
      IndexSpaceNode* shard_node =
          ((launch_space == shard_space) || !shard_space.exists()) ?
              launch_node :
              runtime->get_node(shard_space);
      const DistributedID future_map_did = repl_ctx->get_next_distributed_id();
      // Make a replicate future map
      return repl_ctx->shard_manager->deduplicate_future_map_creation(
          repl_ctx, this, launch_node, shard_node, future_map_did,
          get_provenance());
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finalize_concurrent_mapped(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(concurrent_task);
      legion_assert(!must_epoch_task);
      legion_assert(concurrent_mapping_rendezvous != nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      if (is_recording())
      {
        if (concurrent_functor == 0)
        {
          // The built-in concurrent coloring functor makes this easy
          // since it always maps all the points to the same color
          // See if we're the shard that owns the first point in the
          // launch, if we are then we're the one to make the barrier
          Domain launch_domain = launch_space->get_tight_domain();
          Domain sharding_domain;
          if (sharding_space.exists())
            runtime->find_domain(sharding_space, sharding_domain);
          else
            sharding_domain = launch_domain;
          Domain::DomainPointIterator itr(launch_domain);
          ShardID first_nonempty_shard =
              sharding_function->find_owner(*itr, sharding_domain);
          if (first_nonempty_shard == repl_ctx->owner_shard->shard_id)
          {
            ConcurrentGroup& group = concurrent_groups[0];
            const RtBarrier barrier =
                runtime->create_rt_barrier(group.color_points);
            concurrent_mapping_rendezvous->set_trace_barrier(
                0, barrier, group.color_points);
          }
        }
        else
        {
          // Not the built-in functor so we actually need to some work
          ConcurrentColoringFunctor* functor =
              runtime->find_concurrent_coloring_functor(concurrent_functor);
          // Iterate the whole index domain and find all the points
          // that map to colors that we have on this shard so we can
          // count how many arrivals there are and determine if we're
          // the lowest shard to have a point with this color in which
          // case we're also going to make the barrier for this color
          Domain launch_domain = launch_space->get_tight_domain();
          Domain sharding_domain;
          if (sharding_space.exists())
            runtime->find_domain(sharding_space, sharding_domain);
          else
            sharding_domain = launch_domain;
          for (Domain::DomainPointIterator itr(launch_domain); itr; itr++)
          {
            Color color = functor->color(*itr, index_domain);
            std::map<Color, ConcurrentGroup>::iterator finder =
                concurrent_groups.find(color);
            // Not one of our local colors then we don't care about it
            if (finder == concurrent_groups.end())
              continue;
            finder->second.color_points++;
            // Compute the shard for this point
            ShardID color_shard =
                sharding_function->find_owner(*itr, sharding_domain);
            if (!std::binary_search(
                    finder->second.shards.begin(), finder->second.shards.end(),
                    color_shard))
            {
              finder->second.shards.emplace_back(color_shard);
              std::sort(
                  finder->second.shards.begin(), finder->second.shards.end());
            }
          }
          // See if our local shards is the smallest shard and make and
          // record the barrier if we are
          for (const std::pair<const Color, ConcurrentGroup>& group :
               concurrent_groups)
          {
            legion_assert(!group.second.shards.empty());
            if (group.second.shards.front() == repl_ctx->owner_shard->shard_id)
            {
              const RtBarrier barrier =
                  runtime->create_rt_barrier(group.second.color_points);
              concurrent_mapping_rendezvous->set_trace_barrier(
                  group.first, barrier, group.second.color_points);
            }
          }
        }
      }
      concurrent_mapping_rendezvous->perform_rendezvous();
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finish_concurrent_mapped(
        const std::map<Color, std::pair<RtBarrier, size_t>>& trace_barriers)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      for (std::pair<const Color, ConcurrentGroup>& group : concurrent_groups)
      {
        // Skip groups that we don't have any local shards for
        if (group.second.group_points == 0)
          continue;
        legion_assert(group.second.precondition.interpreted.exists());
        legion_assert(group.second.exchange == nullptr);
        if (is_recording())
        {
          std::map<Color, std::pair<RtBarrier, size_t>>::const_iterator finder =
              trace_barriers.find(group.first);
          legion_assert(finder != trace_barriers.end());
          legion_assert(finder->second.second == group.second.color_points);
          tpl->record_concurrent_group(
              this, group.first, group.second.group_points,
              group.second.color_points, finder->second.first,
              group.second.shards);
        }
        std::map<Color, CollectiveID>::const_iterator finder =
            concurrent_exchange_ids.find(group.first);
        legion_assert(finder != concurrent_exchange_ids.end());
        // Create the max allreduce collective
        group.second.exchange = new ConcurrentAllreduce(
            repl_ctx, finder->second, group.first, group.second.shards);
        if (!group.second.preconditions.empty())
          Runtime::trigger_event(
              group.second.precondition.interpreted,
              Runtime::merge_events(group.second.preconditions));
        else
          Runtime::trigger_event(group.second.precondition.interpreted);
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::initialize_concurrent_group(
        Color color, size_t local, size_t global, RtBarrier barrier,
        const std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      IndexTask::initialize_concurrent_group(
          color, local, global, barrier, shards);
      // Create a concurrent exchange for this color
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
      std::map<Color, CollectiveID>::const_iterator id_finder =
          concurrent_exchange_ids.find(color);
      legion_assert(finder != concurrent_groups.end());
      legion_assert(finder->second.exchange == nullptr);
      legion_assert(id_finder != concurrent_exchange_ids.end());
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      finder->second.shards = shards;
      finder->second.exchange = new ConcurrentAllreduce(
          repl_ctx, id_finder->second, color, finder->second.shards);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::concurrent_allreduce(
        Color color, SliceTask* slice, AddressSpaceID slice_space,
        size_t points, uint64_t lamport_clock, VariantID vid, bool poisoned)
    //--------------------------------------------------------------------------
    {
      if (must_epoch_task)
      {
        legion_assert(color == 0);
        must_epoch->concurrent_allreduce(
            slice, slice_space, points, lamport_clock, poisoned);
        return;
      }
      bool done = false;
      std::map<Color, ConcurrentGroup>::iterator finder =
          concurrent_groups.find(color);
      legion_assert(finder != concurrent_groups.end());
      {
        AutoLock o_lock(op_lock);
        if (finder->second.lamport_clock < lamport_clock)
          finder->second.lamport_clock = lamport_clock;
        if (poisoned)
          finder->second.poisoned = true;
        if (finder->second.slice_tasks.empty())
          finder->second.variant = vid;
        else if (finder->second.variant != vid)
          finder->second.variant = std::min(finder->second.variant, vid);
        finder->second.slice_tasks.emplace_back(
            std::make_pair(slice, slice_space));
        legion_assert(points <= finder->second.group_points);
        finder->second.group_points -= points;
        done = (finder->second.group_points == 0);
      }
      if (done)
      {
        if (finder->second.variant > 0)
        {
          VariantImpl* variant =
              runtime->find_variant_impl(task_id, finder->second.variant);
          if (variant->needs_barrier())
          {
            ReplicateContext* repl_ctx =
                legion_safe_cast<ReplicateContext*>(parent_ctx);
            legion_assert(!finder->second.shards.empty());
            if (finder->second.shards.front() ==
                repl_ctx->owner_shard->shard_id)
              finder->second.task_barrier =
                  runtime->create_rt_barrier(finder->second.color_points);
          }
        }
        finder->second.exchange->perform_concurrent_allreduce(finder->second);
      }
    }

    //--------------------------------------------------------------------------
    uint64_t ReplIndexTask::collective_lamport_allreduce(
        uint64_t lamport_clock, size_t points, bool need_result)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      {
        AutoLock o_lock(op_lock);
        if (collective_lamport_clock < lamport_clock)
          collective_lamport_clock = lamport_clock;
        collective_unbounded_points += points;
        legion_assert(collective_unbounded_points <= total_points);
        if (collective_unbounded_points < total_points)
        {
          if (need_result)
          {
            if (collective_exchange == nullptr)
              collective_exchange =
                  new AllReduceCollective<MaxReduction<uint64_t>, false>(
                      repl_ctx, collective_exchange_id);
            o_lock.release();
            collective_exchange->get_done_event().wait();
            return collective_exchange->get_result();
          }
          else
            return collective_lamport_clock;
        }
        // Otherwise we're going to fall through and do the allreduce
        if (collective_exchange == nullptr)
          collective_exchange =
              new AllReduceCollective<MaxReduction<uint64_t>, false>(
                  repl_ctx, collective_exchange_id);
        if (!need_result)
          commit_preconditions.insert(collective_exchange->get_done_event());
      }
      collective_exchange->async_all_reduce(collective_lamport_clock);
      if (need_result)
      {
        collective_exchange->get_done_event().wait();
        return collective_exchange->get_result();
      }
      else
        return collective_lamport_clock;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplIndexTask::find_pointwise_dependence(
        const DomainPoint& point, GenerationID needed_gen, bool intra_space,
        RtUserEvent to_trigger, std::optional<unsigned> output_region)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      legion_assert(needed_gen <= gen);
      if ((needed_gen < gen) || mapped ||
          (predication_state == PREDICATED_FALSE_STATE))
      {
        if (to_trigger.exists())
          Runtime::trigger_event(to_trigger);
        return RtEvent::NO_RT_EVENT;
      }
      legion_assert(sharding_function != nullptr);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      Domain launch_domain;
      if (sharding_space.exists())
        runtime->find_domain(sharding_space, launch_domain);
      else
        launch_domain = launch_space->get_tight_domain();
      const ShardID point_shard =
          sharding_function->find_owner(point, launch_domain);
      if (point_shard != repl_ctx->owner_shard->shard_id)
      {
        o_lock.release();
        // Send it off to the remote shard to find it
        return repl_ctx->find_pointwise_dependence(
            context_index, point, point_shard, intra_space, to_trigger);
      }
      RtEvent point_mapped;
      // See if we can find this in the point mapped events data structure
      std::map<DomainPoint, RtEvent>::const_iterator finder =
          point_mapped_events.find(point);
      if (finder == point_mapped_events.end())
      {
        // Create a pending pointwise dependence for this point
        std::map<DomainPoint, RtUserEvent>::const_iterator pending_finder =
            pending_pointwise_dependences.find(point);
        if (pending_finder == pending_pointwise_dependences.end())
        {
          RtUserEvent pending = Runtime::create_rt_user_event();
          pending_pointwise_dependences.emplace(std::make_pair(point, pending));
          point_mapped = pending;
        }
        else
          point_mapped = pending_finder->second;
      }
      else
        point_mapped = finder->second;
      // If this is an intra-space look-up or we don't have output regions
      // then we are already done
      if (intra_space || output_pointwise_dependences.empty())
      {
        legion_assert(!output_region);
        if (to_trigger.exists())
        {
          Runtime::trigger_event(to_trigger, point_mapped);
          return to_trigger;
        }
        else
          return point_mapped;
      }
      // Now for the hairy part: if we have globally indexed output regions then
      // we also need to compute transitive dependences for each of the points
      // that this next point will depend on as well in order to run safely
      std::vector<RtEvent> preconditions(1, point_mapped);
      for (unsigned idx = 0; idx < output_pointwise_dependences.size(); idx++)
      {
        if (!output_region_options[idx].global_indexing())
          continue;
        if (output_region && (idx != *output_region))
          continue;
        std::map<DomainPoint, RtEvent>& output_pointwise =
            output_pointwise_dependences[idx];
        DomainPoint color_point = point;
        IndexPartNode* partition = nullptr;
        ProjectionFunction* projection = nullptr;
        const RegionRequirement& req = output_regions[idx];
        const OutputRegionState& state = output_region_states[idx];
        if (req.projection != 0)
        {
          // Need to convert the point into the color space to check if we
          // have it already
          partition = runtime->get_node(req.partition.get_index_partition());
          projection = runtime->find_projection_function(req.projection);
          LogicalRegion region = projection->project_point(
              this, regions.size() + idx, index_domain, point);
          legion_assert(region.exists());
          color_point = runtime->get_node(region.get_index_space())
                            ->get_domain_point_color();
          legion_assert(state.global_color_space.contains(color_point));
        }
        finder = output_pointwise.find(color_point);
        if (finder == output_pointwise.end())
        {
          // DFS backwards for this output region to find the points we need
          std::vector<DomainPoint> needed_colors(1, color_point);
          while (!needed_colors.empty())
          {
            const DomainPoint& needed_color = needed_colors.back();
            std::vector<RtEvent> point_preconditions;
            // Check to see if we have all our previous points
            bool has_all_previous = true;
            for (int dim = 0; dim < needed_color.get_dim(); dim++)
            {
              DomainPoint previous = needed_color;
              previous[dim]--;
              if (!state.global_color_space.contains(previous))
                continue;
              finder = output_pointwise.find(previous);
              if (finder == output_pointwise.end())
              {
                needed_colors.push_back(previous);
                has_all_previous = false;
                break;
              }
              point_preconditions.push_back(finder->second);
            }
            if (!has_all_previous)
              continue;
            // Also include the particular point's mapped event
            DomainPoint orig_point = needed_color;
            if (projection != nullptr)
            {
              // Need to convert from color space back to the index space
              IndexSpaceNode* child = partition->get_child(
                  partition->color_space->linearize_color(needed_color));
              LogicalRegion region(
                  req.partition.tree_did, child->handle,
                  req.partition.field_space);
              std::vector<DomainPoint> points;
              projection->functor->invert(
                  region, req.partition, index_domain, points);
              if (points.size() != 1)
              {
                Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
                error
                    << "Illegal projection function inversion computed by "
                    << "projection functor " << projection->projection_id
                    << " for output region requirement " << idx << " of "
                    << *this << ". The projection function says that "
                    << points.size()
                    << " different points map to logical region " << region
                    << ", however projection functions for output "
                    << "region requirements are required to be bijiective and "
                    << "and therefore exactly one point should ever be allowed "
                    << "to map to a single output logical region.";
                error.raise();
              }
              orig_point = points.back();
            }
            finder = point_mapped_events.find(orig_point);
            if (finder == point_mapped_events.end())
            {
              // See if it was sharded to us or not
              const ShardID pending_shard =
                  sharding_function->find_owner(orig_point, launch_domain);
              if (pending_shard == repl_ctx->owner_shard->shard_id)
              {
                // Sharded locally so either find it or make a pending
                std::map<DomainPoint, RtUserEvent>::const_iterator
                    pending_finder =
                        pending_pointwise_dependences.find(orig_point);
                if (pending_finder == pending_pointwise_dependences.end())
                {
                  RtUserEvent pending = Runtime::create_rt_user_event();
                  pending_pointwise_dependences.emplace(
                      std::make_pair(orig_point, pending));
                }
                else
                  point_preconditions.push_back(pending_finder->second);
              }
              else
              {
                // Send a request to find the output region pointwise dependence
                // Can't be holding the lock when we do this or we risk deadlock
                // This looks scary, but actually isn't, if we race then we'll
                // just do some extra event mergers and generate two events
                // that mean the same thing
                RtUserEvent pending = Runtime::create_rt_user_event();
                point_mapped_events.emplace(
                    std::make_pair(orig_point, pending));
                o_lock.release();
                repl_ctx->find_pointwise_dependence(
                    context_index, orig_point, pending_shard,
                    false /*intra space*/, pending, idx);
                o_lock.reacquire();
              }
            }
            else
              point_preconditions.push_back(finder->second);
            output_pointwise[needed_color] =
                Runtime::merge_events(point_preconditions);
            needed_colors.pop_back();
          }
          finder = output_pointwise.find(point);
          legion_assert(finder != output_pointwise.end());
        }
        preconditions.push_back(finder->second);
      }
      if (to_trigger.exists())
      {
        Runtime::trigger_event(
            to_trigger, Runtime::merge_events(preconditions));
        return to_trigger;
      }
      else
        return Runtime::merge_events(preconditions);
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::record_output_offset(
        unsigned index, unsigned dim, size_t color_index, size_t offset)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      legion_assert(index < output_region_states.size());
      OutputRegionState& state = output_region_states[index];
      legion_assert(dim < state.offsets.size());
      legion_assert(color_index < state.offsets[dim].size());
      legion_assert(state.offsets[dim][color_index] < 0);
      state.offsets[dim][color_index] = offset;
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::record_output_extent(
        unsigned index, const DomainPoint& color, const DomainPoint& extent)
    //--------------------------------------------------------------------------
    {
      legion_assert(index < output_regions.size());
      legion_assert(index < output_region_states.size());
      legion_assert(output_regions.size() == output_region_options.size());
      legion_assert(!is_output_bounded(index));
      const OutputOptions& options = output_region_options[index];
      legion_assert(!options.bounded_requirement());
      OutputRegionState& state = output_region_states[index];
      // Should always be able to set the domain for the color when we're done
      bool done;
      std::vector<DomainPoint> done_points;
      {
        // Check to see if there are any buffered objects in the context
        // that we need to handle if this is global indexing, must do this
        // before taking the lock to avoid deadlock due to taking locks
        // in the wrong order
        std::vector<std::optional<std::pair<size_t, size_t>>> offsets;
        if (options.global_indexing())
        {
          ReplicateContext* repl_ctx =
              legion_safe_cast<ReplicateContext*>(parent_ctx);
          repl_ctx->find_pending_output_offsets(context_index, index, offsets);
        }
        AutoLock o_lock(op_lock);
        if (!offsets.empty())
        {
          legion_assert(state.points.empty());
          for (unsigned dim = 0; dim < offsets.size(); dim++)
          {
            if (!offsets[dim])
              continue;
            const size_t color_index = offsets[dim]->first;
            const size_t offset = offsets[dim]->second;
            legion_assert(dim < state.offsets.size());
            legion_assert(color_index < state.offsets[dim].size());
            legion_assert(state.offsets[dim][color_index] < 0);
            state.offsets[dim][color_index] = offset;
          }
        }
        if (!state.points.emplace(std::make_pair(color, extent)).second)
        {
          const OutputRequirement& req = output_regions[index];
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error << "A projection functor for every output requirement must "
                << "be bijective, but projection functor " << req.projection
                << " for output requirement " << index << " in task " << *this
                << " mapped more than one point in the launch domain to the "
                << "same subregion of color " << color << ".";
          error.raise();
        }
        if (options.global_indexing())
        {
          if (color == state.global_color_space.hi())
            state.has_hi_point = true;
          unsigned remaining_dims = 0;
          legion_assert(color.get_dim() == extent.get_dim());
          // Iterate over each of the dimensions
          for (int dim = 0; dim < color.get_dim(); dim++)
          {
            const size_t color_index =
                color[dim] - state.global_color_space.lo()[dim];
            std::vector<coord_t>& dim_extents = state.extents[dim];
            legion_assert(color_index < dim_extents.size());
            // Check to see if the offset is ready
            std::vector<coord_t>& dim_offsets = state.offsets[dim];
            legion_assert(color_index < dim_offsets.size());
            if (0 <= dim_extents[color_index])
            {
              // Already set, so there's nothing to do here other than
              // to check that the extent matches what we expect
              if (dim_extents[color_index] != extent[dim])
              {
                Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
                error << "Point task " << color
                      << " returned an output of extent " << extent[dim]
                      << " for dimension " << dim
                      << ", but an adjacent point task returned an output "
                      << "of extent " << dim_extents[dim] << ". Please make "
                      << "sure the outputs from point tasks are aligned.";
                error.raise();
              }
              if (dim_offsets[color_index] < 0)
              {
                remaining_dims++;
                if (state.dim_pending_points.empty())
                  state.dim_pending_points.resize(color.get_dim());
                state.dim_pending_points[dim].insert(color);
              }
              continue;
            }
            else  // We're a new extent, so we can save it
              dim_extents[color_index] = extent[dim];
            // If the previous offset is not ready yet there's nothing to do
            if (dim_offsets[color_index] < 0)
            {
              remaining_dims++;
              if (state.dim_pending_points.empty())
                state.dim_pending_points.resize(color.get_dim());
              state.dim_pending_points[dim].insert(color);
              continue;
            }
            // Compute the set of shards for all the points associated with
            // the current extent
            std::set<ShardID> previous_shards;
            compute_output_extent_shards(
                index, dim, color_index, state.global_color_space,
                state.projection, previous_shards);
            // Ripple carry add the extents forward and send out any messages
            // to other shards that need to be notified
            for (unsigned idx = color_index; idx < dim_extents.size(); idx++)
            {
              if (dim_extents[idx] < 0)
                break;
              dim_offsets[idx + 1] = dim_offsets[idx] + dim_extents[idx];
              // Compute the set of shards associated with the next extent
              if (idx < (dim_extents.size() - 1))
              {
                std::set<ShardID> next_shards;
                compute_output_extent_shards(
                    index, dim, idx + 1, state.global_color_space,
                    state.projection, next_shards);
                send_output_offset_messages(
                    index, dim, idx + 1, dim_offsets[idx + 1], previous_shards,
                    next_shards);
                previous_shards.swap(next_shards);
              }
              // Check for any local points for which we are done now
              if (!state.dim_pending_points.empty() &&
                  !state.dim_pending_points[dim].empty())
              {
                for (std::set<DomainPoint>::iterator it =
                         state.dim_pending_points[dim].begin();
                     it != state.dim_pending_points[dim].end();
                     /*nothing*/)
                {
                  const unsigned index =
                      (*it)[dim] - state.global_color_space.lo()[dim];
                  if (index == (idx + 1))
                  {
                    std::map<DomainPoint, unsigned>::iterator finder =
                        state.pending_points.find(*it);
                    legion_assert(finder != state.pending_points.end());
                    legion_assert(finder->second > 0);
                    if (--finder->second == 0)
                    {
                      done_points.push_back(*it);
                      state.pending_points.erase(finder);
                    }
                    std::set<DomainPoint>::iterator delete_it = it++;
                    state.dim_pending_points[dim].erase(delete_it);
                  }
                  else
                    it++;
                }
              }
            }
          }
          if (remaining_dims > 0)
          {
            legion_assert(
                state.pending_points.find(color) == state.pending_points.end());
            state.pending_points.emplace(std::make_pair(color, remaining_dims));
          }
          else
            done_points.push_back(color);
        }
        done = (state.points.size() == total_points);
        if (done)
        {
          // Check to see if all the other output regions are done too
          for (unsigned idx = 0; idx < output_regions.size(); idx++)
          {
            if (output_region_states[idx].points.size() == total_points)
              continue;
            done = false;
            break;
          }
        }
      }
      if (!done_points.empty())
      {
        IndexPartNode* part = runtime->get_node(
            output_regions[index].partition.get_index_partition());
        for (std::vector<DomainPoint>::const_iterator it = done_points.begin();
             it != done_points.end(); it++)
        {
          DomainPoint lo, hi;
          lo.dim = color.dim;
          hi.dim = color.dim;
          for (int dim = 0; dim < color.dim; dim++)
          {
            const unsigned color_index =
                (*it)[dim] - state.global_color_space.lo()[dim];
            lo[dim] = state.offsets[dim][color_index];
            hi[dim] = state.offsets[dim][color_index + 1] - 1;  // inclusive
          }
          IndexSpaceNode* child =
              part->get_child(part->color_space->linearize_color(*it));
          if (child->set_domain(
                  Domain(lo, hi), ApEvent::NO_AP_EVENT,
                  false /*take ownership*/, true /*broadcast*/))
            delete child;
        }
      }
      if (done)
      {
        // Still need to participate in the exchange for computing the
        // top-level index space result regardless of indexing mode
        finalize_output_regions(true /*first invocation*/);
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::compute_output_extent_shards(
        unsigned index, unsigned dim, size_t color_index,
        const Domain& color_space, ProjectionFunction* projection,
        std::set<ShardID>& shards) const
    //--------------------------------------------------------------------------
    {
      legion_assert(shards.empty());
      legion_assert(sharding_function != nullptr);
      DomainPoint lo = color_space.lo();
      DomainPoint hi = color_space.hi();
      // Restrict to all the points that shards the same color index on the dim
      lo[dim] += color_index;
      legion_assert(lo[dim] <= hi[dim]);
      hi[dim] = lo[dim];
      const Domain next_colors(lo, hi);
      Domain sharding_domain;
      if (sharding_space.exists())
        runtime->find_domain(sharding_space, sharding_domain);
      else
        sharding_domain = index_domain;
      if (projection->projection_id == 0)
      {
        // Identity projection is trivially invertible
        for (Domain::DomainPointIterator itr(next_colors); itr; itr++)
        {
          const ShardID shard =
              sharding_function->find_owner(*itr, sharding_domain);
          shards.insert(shard);
        }
      }
      else
      {
        // Need to invert to points in the launch domain before we can
        // do the sharding computation
        std::vector<LogicalRegion> next_regions;
        next_regions.reserve(next_colors.get_volume());
        const RegionRequirement& req = output_regions[index];
        IndexPartNode* partition =
            runtime->get_node(req.partition.get_index_partition());
        for (Domain::DomainPointIterator itr(next_colors); itr; itr++)
        {
          LegionColor color = partition->color_space->linearize_color(*itr);
          // This is a bit ineffecient because we really only need the name of
          // child and don't need to actually construct the node for it here in
          // order to do that but it does help in cases where we'll need to look
          // it up multiple times so we'll end up caching the result so we'll
          // just leave it for now
          IndexSpaceNode* child = partition->get_child(color);
          next_regions.emplace_back(LogicalRegion(
              req.partition.tree_did, child->handle,
              req.partition.field_space));
        }
        std::map<LogicalRegion, std::vector<DomainPoint>> next_points;
        projection->find_inversions(
            TASK_OP_KIND, unique_op_id, regions.size() + index, req,
            launch_space, next_regions, next_points);
        for (std::map<LogicalRegion, std::vector<DomainPoint>>::const_iterator
                 it = next_points.begin();
             it != next_points.end(); it++)
        {
          if (it->second.size() != 1)
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Illegal projection function inversion computed by "
                  << "projection functor " << projection->projection_id
                  << " for output region requirement " << index << " of "
                  << *this << ". The projection function says that "
                  << it->second.size()
                  << " different points map to logical region " << it->first
                  << ", however projection functions for output "
                  << "region requirements are required to be bijiective and "
                  << "and therefore exactly one point should ever be allowed "
                  << "to map to a single output logical region.";
            error.raise();
          }
          const ShardID shard =
              sharding_function->find_owner(it->second.back(), sharding_domain);
          shards.insert(shard);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::send_output_offset_messages(
        unsigned index, unsigned dim, size_t color_index, size_t offset,
        const std::set<ShardID>& previous_shards,
        const std::set<ShardID>& next_shards) const
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      ShardManager* shard_manager = repl_ctx->shard_manager;
      const ShardID local_shard = repl_ctx->owner_shard->shard_id;
      legion_assert(previous_shards.find(local_shard) != previous_shards.end());
      // For each of the next shards, see if we need to send them a message
      // with the output extent for this color_index of this dim
      for (std::set<ShardID>::const_iterator it = next_shards.begin();
           it != next_shards.end(); it++)
      {
        std::set<ShardID>::const_iterator finder =
            previous_shards.lower_bound(*it);
        if (finder == previous_shards.end())
        {
          // Above the last previous shard
          // If we're not the last previous shard then we don't send it
          if (local_shard != *std::prev(previous_shards.end()))
            continue;
        }
        else if ((*finder) == *it)
        {
          // Shard will provide the extent to itself
          continue;
        }
        else if (finder == previous_shards.begin())
        {
          // Before the first previous shard
          // If we're not the first previous_shard then don't send it
          if (local_shard != *previous_shards.begin())
            continue;
        }
        else
        {
          // Between two entries
          std::set<ShardID>::const_iterator previous = std::prev(finder);
          if ((*it - *previous) <= (*finder - *it))
          {
            // Previous is closer
            if (local_shard != *previous)
              continue;
          }
          else
          {
            // Finder is closer
            if (local_shard != *finder)
              continue;
          }
        }
        // If we get here then we're sending the message to the next shard
        ReplOutputOffset rez;
        rez.serialize(shard_manager->did);
        rez.serialize(*it);
        rez.serialize(context_index);
        rez.serialize(index);
        rez.serialize(dim);
        rez.serialize(color_index);
        rez.serialize(offset);
        shard_manager->send_output_offset(*it, rez);
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::validate_output_extents(unsigned index)
    //--------------------------------------------------------------------------
    {
      IndexTask::validate_output_extents(index);
      // Since we don't see all the points on any one shard we still need to
      // check for any global indexing violations where points on the same
      // dimension given different extents.
      if (output_region_options[index].bounded_requirement())
        return;
      if (!output_region_options[index].global_indexing())
        return;
      OutputRegionState& state = output_region_states[index];
      for (std::map<DomainPoint, DomainPoint>::const_iterator it =
               state.points.begin();
           it != state.points.end(); it++)
      {
        legion_assert(it->first.get_dim() == it->second.get_dim());
        for (int dim = 0; dim < it->first.get_dim(); dim++)
        {
          const size_t color_index = it->first[dim];
          legion_assert(((size_t)dim) < state.extents.size());
          legion_assert(color_index < state.extents[dim].size());
          if (state.extents[dim][color_index] < 0)
            state.extents[dim][color_index] = it->second[dim];
          else if (state.extents[dim][color_index] != it->second[dim])
          {
            Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
            error << "Point task " << it->first
                  << " returned an output of extent " << it->second[dim]
                  << " for dimension " << dim << " of output region " << index
                  << ", but an adjacent point task returned an output "
                  << "of extent " << state.extents[dim][color_index]
                  << ". Please make sure the outputs from point tasks "
                  << "are aligned.";
            error.raise();
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::record_output_registered(RtEvent registered)
    //--------------------------------------------------------------------------
    {
      legion_assert(output_bar.exists());
      legion_assert(!output_regions.empty());
      // Record it in the set of output events and if we've seen all of them
      // then we can launch the meta-task to do the final regisration
      AutoLock o_lock(op_lock);
      // Can be a no-event if we've been shared off
      if (registered.exists())
        output_preconditions.emplace_back(registered);
      legion_assert(output_preconditions.size() <= total_points);
      if (output_preconditions.size() == total_points)
      {
        // Can only do the registration once all registrations are done
        runtime->phase_barrier_arrive(
            output_bar, 1 /*count*/,
            Runtime::merge_events(output_preconditions));
        // Can only mark the EqKDTree ready once all the points are registered
        FinalizeOutputEqKDTreeArgs args(this);
        registered = runtime->issue_runtime_meta_task(
            args, LG_LATENCY_DEFERRED_PRIORITY, output_bar);
        commit_preconditions.insert(registered);
      }
    }

    //--------------------------------------------------------------------------
    void ReplIndexTask::finalize_output_regions(bool first_invocation)
    //--------------------------------------------------------------------------
    {
      // Check to see if we have an exchange to perform
      if (first_invocation && (output_size_collective != nullptr))
      {
        // We need to gather output region sizes from all the other shards
        // to determine the sizes of globally indexed output regions
        output_size_collective->perform_collective_async();
        // The collective will call us when it is ready but we still need
        // to make sure that we fold the completion event for the
        // collective back into the completion events
        const RtEvent done_event = output_size_collective->get_done_event();
        if (done_event.exists() && !done_event.has_triggered())
        {
          AutoLock o_lock(op_lock);
          // We should still not be complete if we're here
          legion_assert(
              (completed_points < total_points) || (total_points == 0));
          commit_preconditions.insert(done_event);
        }
      }
      else
      {
        ReplicateContext* repl_ctx =
            legion_safe_cast<ReplicateContext*>(parent_ctx);
        const bool first_local_shard =
            repl_ctx->shard_manager->is_first_local_shard(
                repl_ctx->owner_shard);
        for (unsigned idx = 0; idx < output_regions.size(); ++idx)
        {
          const OutputOptions& options = output_region_options[idx];
          if (options.bounded_requirement())
            continue;
          if (runtime->safe_model)
            validate_output_extents(idx);
          if (options.global_indexing())
          {
            const OutputRegionState& state = output_region_states[idx];
            // Check to see if we had the last point in the color space
            if (state.has_hi_point)
            {
              DomainPoint root_lo, root_hi;
              root_lo.dim = state.offsets.size();
              root_hi.dim = state.offsets.size();
              for (int dim = 0; dim < root_lo.get_dim(); dim++)
              {
                legion_assert(state.offsets[dim].back() >= 0);
                root_lo[dim] = 0;
                root_hi[dim] = state.offsets[dim].back() - 1 /*inclusive*/;
              }
              IndexSpaceNode* parent = runtime->get_node(
                  output_regions[idx].parent.get_index_space());
              if (parent->set_domain(
                      Domain(root_lo, root_hi), ApEvent::NO_AP_EVENT,
                      false /*take ownership*/, true /*broadcast*/))
                delete parent;
            }
          }
          else if (first_local_shard)
          {
            IndexSpaceNode* parent =
                runtime->get_node(output_regions[idx].parent.get_index_space());
            if (parent->set_output_union(output_region_states[idx].points))
              delete parent;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    size_t ReplIndexTask::get_collective_points(void) const
    //--------------------------------------------------------------------------
    {
      return runtime->get_node(internal_space)->get_volume();
    }

    //--------------------------------------------------------------------------
    bool ReplIndexTask::find_shard_participants(std::vector<ShardID>& shards)
    //--------------------------------------------------------------------------
    {
      legion_assert(sharding_function != nullptr);
      if (sharding_space.exists())
        return sharding_function->find_shard_participants(
            launch_space, sharding_space, shards);
      else
        return sharding_function->find_shard_participants(
            launch_space, launch_space->handle, shards);
    }

  }  // namespace Internal
}  // namespace Legion
