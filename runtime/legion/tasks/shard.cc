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

#include "legion/tasks/shard.h"
#include "legion/contexts/leaf.h"
#include "legion/contexts/replicate.h"
#include "legion/api/future_impl.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/tracing/automatic.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Shard Task
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ShardTask::ShardTask(
        SingleTask* source, InnerContext* parent, ShardManager* manager,
        ShardID id, Processor proc, VariantID variant)
      : SingleTask(), shard_id(id)
    //--------------------------------------------------------------------------
    {
      legion_assert(proc.address_space() == runtime->address_space);
      SingleTask::activate();
      set_current_proc(proc);  // do this before clone_single_from
      if (source != nullptr)
        clone_single_from(source);
      else
        parent_ctx = parent;
      stealable = false;
      replicate = false;
      shard_manager = manager;
      shard_manager->add_base_resource_ref(SINGLE_TASK_REF);
      selected_variant = variant;
      // If we have any region requirements then they are all collective
      check_collective_regions.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        check_collective_regions[idx] = idx;
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
        log_requirement(unique_op_id, idx, logical_regions[idx]);
    }

    //--------------------------------------------------------------------------
    ShardTask::ShardTask(
        InnerContext* parent, Deserializer& derez, ShardManager* manager,
        ShardID id, Processor proc, VariantID variant)
      : SingleTask(), shard_id(id)
    //--------------------------------------------------------------------------
    {
      legion_assert(proc.address_space() == runtime->address_space);
      SingleTask::activate();
      set_current_proc(proc);
      stealable = false;
      replicate = false;
      parent_ctx = parent;
      shard_manager = manager;
      shard_manager->add_base_resource_ref(SINGLE_TASK_REF);
      selected_variant = variant;
      std::set<RtEvent> ready_events;
      unpack_single_task(derez, ready_events);
      // If we have any region requirements then they are all collective
      check_collective_regions.resize(regions.size());
      for (unsigned idx = 0; idx < regions.size(); idx++)
        check_collective_regions[idx] = idx;
      for (unsigned idx = 0; idx < logical_regions.size(); idx++)
        log_requirement(unique_op_id, idx, logical_regions[idx]);
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
      legion_assert(shard_manager == nullptr);
    }

    //--------------------------------------------------------------------------
    void ShardTask::activate(void)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardTask::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // Set our shard manager to nullptr since we are not supposed to delete it
      if (shard_manager->remove_base_resource_ref(SINGLE_TASK_REF))
        delete shard_manager;
      shard_manager = nullptr;
      SingleTask::deactivate(false /*free*/);
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
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardTask::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardTask::predicate_false(void)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool ShardTask::distribute_task(void)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    RtEvent ShardTask::perform_must_epoch_version_analysis(MustEpochOp* own)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    bool ShardTask::perform_mapping(
        MustEpochOp* must_epoch_owner, const DeferMappingArgs* args)
    //--------------------------------------------------------------------------
    {
      if (!map_all_regions(must_epoch_owner, args))
        return false;
      shard_manager->handle_post_mapped(true /*local*/, get_mapped_event());
      return true;
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_future_size(
        size_t return_type_size, std::set<RtEvent>& applied_events)
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
    void ShardTask::initialize_map_task_input(
        Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
        MustEpochOp* must_epoch_owner)
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
    bool ShardTask::finalize_map_task_output(
        Mapper::MapTaskInput& input, Mapper::MapTaskOutput& output,
        MustEpochOp* must_epoch_owner)
    //--------------------------------------------------------------------------
    {
      if (!SingleTask::finalize_map_task_output(
              input, output, must_epoch_owner))
        return false;
      // This is a replicated task, the mapper isn't allowed to
      // mutate the target_processors from the shard processor
      if ((output.target_procs.size() != 1) ||
          (output.target_procs.front() != input.shard_processor))
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Mapper " << mapper->get_mapper_name()
              << " provided invalid target_processors from call to "
              << "'map_task' for replicated task " << get_task_name()
              << " (UID " << get_unique_id()
              << "). Replicated tasks are only permitted to have one target "
              << "processor and it must be exactly 'input.shard_procesor' as "
              << "that is where this replicated copy of the task has been "
              << "assigned to run by this same mapper.";
        error.raise();
      }
      if (output.chosen_variant != input.shard_variant)
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'map_task' on "
              << "mapper " << *mapper << ". Mapper specified an invalid task "
              << "variant of ID " << output.chosen_variant
              << " for replicated task " << *this << ", which "
              << "differs from the specified 'input.shard_variant' "
              << input.shard_variant << " previously chosen by the mapper "
              << "in 'replicate_task'. The mapper is required to maintain the "
              << "previously selected variant in the output of 'map_task'.";
        error.raise();
      }
      if (!is_leaf() && !regions.empty() && runtime->safe_mapper)
      {
        legion_assert(mapper != nullptr);
        legion_assert(regions.size() == virtual_mapped.size());
        // If this is not a leaf shard then check that all the shards agree
        // on which regions are going to be virtually mapped and which aren't
        shard_manager->rendezvous_check_virtual_mappings(
            shard_id, mapper, virtual_mapped);
      }
      return true;
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
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardTask::trigger_complete(ApEvent effects)
    //--------------------------------------------------------------------------
    {
      // Invalidate the logical context so child operations that still have
      // mapping references can begin committing
      if (execution_context != nullptr)
        execution_context->invalidate_logical_context();
      // First do the normal clean-up operations
      // Remove profiling our guard and trigger the profiling event if necessary
      if ((outstanding_profiling_requests.fetch_sub(1) == 1) &&
          profiling_reported.exists())
        Runtime::trigger_event(profiling_reported);
      complete_operation(
          shard_manager->trigger_task_complete(true /*local*/, effects));
    }

    //--------------------------------------------------------------------------
    void ShardTask::trigger_task_commit(void)
    //--------------------------------------------------------------------------
    {
      // Commit this operation
      // Invalidate any context that we had so that the child
      // operations can begin committing
      std::set<RtEvent> commit_preconditions;
      execution_context->invalidate_region_tree_contexts(
          is_top_level_task(), commit_preconditions,
          &shard_manager->get_mapping(), shard_id);
      execution_context->log_created_requirements();
      if (profiling_reported.exists() && !profiling_reported.has_triggered())
        commit_preconditions.insert(profiling_reported);
      RtEvent commit_precondition;
      if (!commit_preconditions.empty())
        commit_precondition = Runtime::merge_events(commit_preconditions);
      // Dont' deactivate ourselves, the shard manager will do that for us
      commit_operation(false /*deactivate*/);
      // Lastly invoke the method on the shard manager, this could
      // delete us so it has to be last
      shard_manager->trigger_task_commit(true /*local*/, commit_precondition);
    }

    //--------------------------------------------------------------------------
    bool ShardTask::send_task(
        Processor target, std::vector<SingleTask*>& others)
    //--------------------------------------------------------------------------
    {
      Error error(LEGION_MAPPER_EXCEPTION);
      error << "Mapper " << mapper->get_mapper_name()
            << " requested that shard task " << get_task_name() << " (UID "
            << get_unique_id()
            << ") be moved to a remote node. Shard tasks must be mapped to "
            << "the processors assigned by replicate_task and therefore cannot "
            << "be moved to a remote node.";
      error.raise();
      return false;
    }

    //--------------------------------------------------------------------------
    bool ShardTask::pack_task(Serializer& rez, AddressSpaceID target)
    //--------------------------------------------------------------------------
    {
      RezCheck z(rez);
      pack_single_task(rez, target);
      parent_ctx->pack_inner_context(rez);
      return false;
    }

    //--------------------------------------------------------------------------
    bool ShardTask::unpack_task(
        Deserializer& derez, Processor current, AddressSpaceID,
        std::set<RtEvent>& ready_events)
    //--------------------------------------------------------------------------
    {
      DerezCheck z(derez);
      legion_assert(!single_task_termination.exists());
      unpack_single_task(derez, ready_events);
      parent_ctx = InnerContext::unpack_inner_context(derez);
      set_current_proc(current);
      return false;
    }

    //--------------------------------------------------------------------------
    void ShardTask::perform_inlining(
        VariantImpl* variant, const std::deque<InstanceSet>& parent_regions)
    //--------------------------------------------------------------------------
    {
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_future(
        ApEvent effects, FutureInstance* instance, const void* metadata,
        size_t metasize, FutureFunctor* functor, Processor future_proc,
        bool own_functor)
    //--------------------------------------------------------------------------
    {
      legion_assert(functor == nullptr);
      if ((instance != nullptr) && (instance->size > 0))
        check_future_return_bounds(instance);
      if (shard_manager->handle_future(effects, instance, metadata, metasize) &&
          (instance != nullptr) && !instance->defer_deletion(effects))
        delete instance;
    }

    //--------------------------------------------------------------------------
    void ShardTask::handle_mispredication(void)
    //--------------------------------------------------------------------------
    {
      // TODO: figure out how mispredication works with control replication
      std::abort();
    }

    //--------------------------------------------------------------------------
    uint64_t ShardTask::order_collectively_mapped_unbounded_pools(
        uint64_t lamport_clock, bool need_result)
    //--------------------------------------------------------------------------
    {
      if (is_leaf())
      {
        // If we're a leaf that means we're normally replicated and shards
        // can have point tasks, but they are collectively mapping so we do
        // need to do the lamport all-reduce to make sure we don't deadlock.
        // TODO: implelement this
        std::abort();
      }
      else
      {
        // If we're a non-leaf variant that means we're control replicated
        // and therefore non of the tasks will have pool memories
        return lamport_clock;
      }
    }

    //--------------------------------------------------------------------------
    void ShardTask::concurrent_allreduce(
        ProcessorManager* manager, uint64_t lamport_clock, VariantID vid,
        bool poisoned)
    //--------------------------------------------------------------------------
    {
      // Should never be called
      std::abort();
    }

    //--------------------------------------------------------------------------
    void ShardTask::perform_concurrent_task_barrier(void)
    {
      Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
      error
          << "Illegal concurrent task barrier performed in replicated task "
          << get_task_name() << " (UID " << get_unique_id()
          << "). Concurrent task barriers are not permitted in "
          << "replicated tasks. They can only be performed in concurrent index "
          << "space tasks.";
      error.raise();
    }

    //--------------------------------------------------------------------------
    RtEvent ShardTask::convert_collective_views(
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
      return shard_manager->convert_collective_views(
          requirement_index, analysis_index, region, targets, physical_ctx,
          analysis_mapping, first_local, target_views, collective_arrivals);
    }

    //--------------------------------------------------------------------------
    RtEvent ShardTask::perform_collective_versioning_analysis(
        unsigned index, LogicalRegion handle, EqSetTracker* tracker,
        const FieldMask& mask, unsigned parent_req_index)
    //--------------------------------------------------------------------------
    {
      return shard_manager->rendezvous_collective_versioning_analysis(
          index, handle, tracker, runtime->address_space, mask,
          parent_req_index);
    }

    //--------------------------------------------------------------------------
    TaskContext* ShardTask::create_execution_context(
        VariantImpl* v, std::set<ApEvent>& launch_events, bool inline_task,
        bool leaf_task)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_shard(
          LEGION_DISTRIBUTED_ID_FILTER(shard_manager->did), shard_id,
          get_unique_id());
      if (!leaf_task)
      {
        // Should have checked that we don't have any output regions here
        legion_assert(output_regions.empty());
        legion_assert(virtual_mapped.size() == regions.size());
        legion_assert(parent_req_indexes.size() == regions.size());
        // If we have a control replication context then we do the special path.
        const Mapper::ContextConfigOutput& configuration =
            shard_manager->context_configuration;
        ReplicateContext* repl_ctx = nullptr;
        if (configuration.auto_tracing_enabled)
        {
          log_auto_trace.info(
              "Initializing auto tracing for %s (UID %lld)", get_task_name(),
              get_unique_id());
          repl_ctx = new AutoTracing<ReplicateContext>(
              configuration, this, get_depth(), v->is_inner(), regions,
              output_regions, parent_req_indexes, virtual_mapped, task_priority,
              execution_fence_event, shard_manager, inline_task,
              parent_ctx->is_concurrent_context());
        }
        else
          repl_ctx = new ReplicateContext(
              configuration, this, get_depth(), v->is_inner(), regions,
              output_regions, parent_req_indexes, virtual_mapped, task_priority,
              execution_fence_event, shard_manager, inline_task,
              parent_ctx->is_concurrent_context());
        repl_ctx->add_base_gc_ref(SINGLE_TASK_REF);
        // Save the execution context early since we'll need it
        execution_context = repl_ctx;
        // Make sure that none of the shards start until all the replicate
        // contexts have been made across all the shards
        RtEvent ready = shard_manager->complete_startup_initialization();
        if (ready.exists())
          launch_events.insert(ApEvent(ready));
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
        execution_context =
            new LeafContext(this, std::move(leaf_memory_pools), inline_task);
        execution_context->add_base_gc_ref(SINGLE_TASK_REF);
        leaf_memory_pools.clear();
      }
      return execution_context;
    }

    //--------------------------------------------------------------------------
    InnerContext* ShardTask::create_implicit_context(void)
    //--------------------------------------------------------------------------
    {
      LegionSpy::log_shard(
          LEGION_DISTRIBUTED_ID_FILTER(shard_manager->did), shard_id,
          get_unique_id());
      const Mapper::ContextConfigOutput& configuration =
          shard_manager->context_configuration;
      ReplicateContext* repl_ctx = nullptr;
      if (configuration.auto_tracing_enabled)
      {
        log_auto_trace.info(
            "Initializing auto tracing for %s (UID %lld)", get_task_name(),
            get_unique_id());
        repl_ctx = new AutoTracing<ReplicateContext>(
            configuration, this, get_depth(), false /*is inner*/, regions,
            output_regions, parent_req_indexes, virtual_mapped, task_priority,
            execution_fence_event, shard_manager, false /*inline task*/,
            true /*implicit*/);
      }
      else
        repl_ctx = new ReplicateContext(
            configuration, this, get_depth(), false /*is inner*/, regions,
            output_regions, parent_req_indexes, virtual_mapped, task_priority,
            execution_fence_event, shard_manager, false /*inline task*/,
            true /*implicit*/);
      repl_ctx->add_base_gc_ref(SINGLE_TASK_REF);
      // Save the execution context early since we'll need it
      execution_context = repl_ctx;
      // Implicit don't have variants so we have to set these properties
      // here, implicit top-level tasks are never leaf or inner tasks
      is_leaf_result = false;
      leaf_cached = true;
      is_inner_result = false;
      inner_cached = true;
      // Wait until all the other shards are ready too
      const RtEvent wait_on = shard_manager->complete_startup_initialization();
      if (!wait_on.has_triggered())
        wait_on.wait();
      return repl_ctx;
    }

    //--------------------------------------------------------------------------
    void ShardTask::dispatch(void)
    //--------------------------------------------------------------------------
    {
      // Have to launch a task to do this in case they need to rendezvous
      defer_perform_mapping(
          RtEvent::NO_RT_EVENT, nullptr /*must epoch*/, 0 /*invocation count*/);
    }

    //--------------------------------------------------------------------------
    void ShardTask::return_resources(
        ResourceTracker* target, std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      legion_assert(execution_context != nullptr);
      execution_context->return_resources(target, context_index, preconditions);
    }

    //--------------------------------------------------------------------------
    void ShardTask::report_leaks_and_duplicates(
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      legion_assert(execution_context != nullptr);
      execution_context->report_leaks_and_duplicates(preconditions);
    }

    //--------------------------------------------------------------------------
    ReplicateContext* ShardTask::get_replicate_context(void) const
    //--------------------------------------------------------------------------
    {
      legion_assert(execution_context != nullptr);
      return legion_safe_cast<ReplicateContext*>(execution_context);
    }

    //--------------------------------------------------------------------------
    void ShardTask::initialize_implicit_task(
        TaskID tid, MapperID mid, Processor proxy)
    //--------------------------------------------------------------------------
    {
      task_id = tid;
      map_id = mid;
      orig_proc = proxy;
      current_proc = proxy;
      shard_manager->handle_post_mapped(true /*local*/, get_mapped_event());
    }

  }  // namespace Internal
}  // namespace Legion
