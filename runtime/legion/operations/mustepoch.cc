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

#include "legion/operations/mustepoch.h"
#include "legion/contexts/replicate.h"
#include "legion/managers/mapper.h"
#include "legion/managers/shard.h"
#include "legion/nodes/index.h"
#include "legion/operations/internal.h"
#include "legion/tasks/index.h"
#include "legion/tasks/individual.h"
#include "legion/tasks/point.h"
#include "legion/tasks/slice.h"
#include "legion/utilities/provenance.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Must Epoch Operation
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochOp::MustEpochOp(void) : Operation(), MustEpoch()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MustEpochOp::~MustEpochOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    UniqueID MustEpochOp::get_unique_id(void) const
    //--------------------------------------------------------------------------
    {
      return unique_op_id;
    }

    //--------------------------------------------------------------------------
    uint64_t MustEpochOp::get_context_index(void) const
    //--------------------------------------------------------------------------
    {
      return context_index;
    }

    //--------------------------------------------------------------------------
    int MustEpochOp::get_depth(void) const
    //--------------------------------------------------------------------------
    {
      return (parent_ctx->get_depth() + 1);
    }

    //--------------------------------------------------------------------------
    const Task* MustEpochOp::get_parent_task(void) const
    //--------------------------------------------------------------------------
    {
      if (parent_task == nullptr)
        parent_task = parent_ctx->get_task();
      return parent_task;
    }

    //--------------------------------------------------------------------------
    const std::string_view& MustEpochOp::get_provenance_string(bool human) const
    //--------------------------------------------------------------------------
    {
      Provenance* provenance = get_provenance();
      if (provenance != nullptr)
        return human ? provenance->human : provenance->machine;
      else
        return Provenance::no_provenance;
    }

    //--------------------------------------------------------------------------
    FutureMap MustEpochOp::initialize(
        InnerContext* ctx, const MustEpochLauncher& launcher,
        Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      // Initialize this operation
      initialize_operation(ctx, provenance);
      // Compute our launch domain if we need it
      launch_domain = launcher.launch_domain;
      IndexSpace launch_space = launcher.launch_space;
      if (!launch_space.exists())
      {
        if (!launch_domain.exists())
          launch_space = compute_launch_space(launcher, provenance);
        else
          launch_space =
              ctx->find_index_launch_space(launch_domain, provenance);
      }
      if (!launch_domain.exists())
      {
        runtime->find_domain(launcher.launch_space, launch_domain);
        legion_assert(launch_domain.exists());
      }
      // Make a new future map for storing our results
      // We'll fill it in later
      sharding_space = launcher.sharding_space;
      result_map = create_future_map(ctx, launch_space, sharding_space);
      instantiate_tasks(ctx, launcher);
      map_id = launcher.map_id;
      tag = launcher.mapping_tag;
      parent_task = ctx->get_task();
      if (ctx->is_concurrent_context())
      {
        Fatal fatal;
        fatal << "Illegal nested must epoch launch which has a concurrent "
              << "ancesstor (must epoch or concurrent index task). Nested "
              << "concurrency is not currently supported.";
        fatal.raise();
      }
      LegionSpy::log_must_epoch_operation(ctx->get_unique_id(), unique_op_id);
      return result_map;
    }

    //--------------------------------------------------------------------------
    FutureMap MustEpochOp::create_future_map(
        TaskContext* ctx, IndexSpace domain, IndexSpace shard_space)
    //--------------------------------------------------------------------------
    {
      IndexSpaceNode* launch_node = runtime->get_node(domain);
      return FutureMap(new FutureMapImpl(
          ctx, this, launch_node, runtime->get_available_distributed_id(),
          get_provenance()));
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::instantiate_tasks(
        InnerContext* ctx, const MustEpochLauncher& launcher)
    //--------------------------------------------------------------------------
    {
      legion_assert(!concurrent_mapped.exists());
      concurrent_mapped = Runtime::create_rt_user_event();
      // Initialize operations for everything in the launcher
      // Note that we do not track these operations as we want them all to
      // appear as a single operation to the parent context in order to
      // avoid deadlock with the maximum window size.
      indiv_tasks.resize(launcher.single_tasks.size());
      Provenance* provenance = get_provenance();
      for (unsigned idx = 0; idx < launcher.single_tasks.size(); idx++)
      {
        indiv_tasks[idx] = runtime->get_operation<IndividualTask>();
        indiv_tasks[idx]->initialize_task(
            ctx, launcher.single_tasks[idx], provenance, false /*top level*/,
            true /*must epoch*/);
        indiv_tasks[idx]->set_must_epoch(this, idx, true /*register*/);
        indiv_tasks[idx]->set_concurrent_postcondition(concurrent_mapped);
        // If we have a trace, set it for this operation as well
        if (trace != nullptr)
          trace->initialize_operation(indiv_tasks[idx], nullptr);
      }
      index_tasks.resize(launcher.index_tasks.size());
      for (unsigned idx = 0; idx < launcher.index_tasks.size(); idx++)
      {
        if (launcher.index_tasks[idx].concurrent_functor != 0)
        {
          Error error(LEGION_INTERFACE_EXCEPTION);
          error << "All index space task launches in must epoch operations "
                << "are required to use default concurrent functor (ID=0).";
          error.raise();
        }
        IndexSpace launch_space = launcher.index_tasks[idx].launch_space;
        if (!launch_space.exists())
          launch_space = ctx->find_index_launch_space(
              launcher.index_tasks[idx].launch_domain, get_provenance());
        index_tasks[idx] = runtime->get_operation<IndexTask>();
        index_tasks[idx]->initialize_task(
            ctx, launcher.index_tasks[idx], launch_space, provenance,
            false /*track*/);
        index_tasks[idx]->set_must_epoch(
            this, indiv_tasks.size() + idx, true /*register*/);
        index_tasks[idx]->initialize_must_epoch_concurrent_group(
            0 /*color*/, concurrent_mapped);
        if (trace != nullptr)
          trace->initialize_operation(index_tasks[idx], nullptr);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::find_conflicted_regions(
        std::vector<PhysicalRegion>& conflicts)
    //--------------------------------------------------------------------------
    {
      // Dump them all into a set when they are done to deduplicate them
      // This is not the most optimized way to do this, but it will work for now
      std::set<PhysicalRegion> temp_conflicts;
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        std::vector<PhysicalRegion> temp;
        parent_ctx->find_conflicting_regions(indiv_tasks[idx], temp);
        temp_conflicts.insert(temp.begin(), temp.end());
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        std::vector<PhysicalRegion> temp;
        parent_ctx->find_conflicting_regions(index_tasks[idx], temp);
        temp_conflicts.insert(temp.begin(), temp.end());
      }
      conflicts.insert(
          conflicts.end(), temp_conflicts.begin(), temp_conflicts.end());
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::activate(void)
    //--------------------------------------------------------------------------
    {
      Operation::activate();
      map_id = 0;
      tag = 0;
      parent_task = nullptr;
      launch_domain = Domain();
      individual_tasks.clear();
      index_space_tasks.clear();
      single_tasks_ready = RtUserEvent::NO_RT_USER_EVENT;
      concurrent_mapped = RtUserEvent::NO_RT_USER_EVENT;
      remaining_mapped_events.store(0);
      remaining_concurrent_mapped = 0;
      remaining_single_tasks.store(0);
      remaining_resource_returns = 0;
      remaining_concurrent_points = 0;
      concurrent_lamport_clock = 0;
      concurrent_poisoned = false;
      remaining_collective_unbound_points = 0;
      collective_lamport_clock = 0;
      collective_lamport_clock_ready = RtUserEvent::NO_RT_USER_EVENT;
      // Set to 1 to include the triggers we get for our operation
      remaining_subop_completes = 0;
      remaining_subop_commits = 1;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      // All the sub-operations we have will deactivate themselves
      indiv_tasks.clear();
      index_tasks.clear();
      slice_tasks.clear();
      single_tasks.clear();
      concurrent_preconditions.clear();
      concurrent_tasks.clear();
      concurrent_slices.clear();
      // Remove our reference on the future map
      result_map = FutureMap();
      task_sets.clear();
      if (!acquired_instances.empty())
        release_acquired_instances(acquired_instances);
      dependence_map.clear();
      for (DependenceRecord*& record : dependences)
      {
        delete record;
      }
      dependences.clear();
      single_task_map.clear();
      mapping_dependences.clear();
      mapped_events.clear();
      input.tasks.clear();
      input.constraints.clear();
      output.task_processors.clear();
      output.constraint_mappings.clear();
      slice_version_events.clear();
      commit_preconditions.clear();
      Operation::deactivate(false /*free*/);
      // Return this operation to the free list
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    const char* MustEpochOp::get_logging_name(void) const
    //--------------------------------------------------------------------------
    {
      return op_names[MUST_EPOCH_OP_KIND];
    }

    //--------------------------------------------------------------------------
    OpKind MustEpochOp::get_operation_kind(void) const
    //--------------------------------------------------------------------------
    {
      return MUST_EPOCH_OP_KIND;
    }

    //--------------------------------------------------------------------------
    size_t MustEpochOp::get_region_count(void) const
    //--------------------------------------------------------------------------
    {
      size_t result = 0;
      for (IndividualTask* const & task : indiv_tasks)
      {
        result += task->get_region_count();
      }
      for (IndexTask* const & task : index_tasks)
      {
        result += task->get_region_count();
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level > NO_SPY_LOGGING)
      {
        for (IndividualTask* const & task : indiv_tasks)
          LegionSpy::log_child_operation_index(
              parent_ctx->get_unique_id(), context_index,
              task->get_unique_op_id());
        for (IndexTask* const & task : index_tasks)
          LegionSpy::log_child_operation_index(
              parent_ctx->get_unique_id(), context_index,
              task->get_unique_op_id());
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_dependence_analysis(void)
    //--------------------------------------------------------------------------
    {
      // Record how many mapping dependences we expect to be notified of
      remaining_mapping_dependences.fetch_add(
          indiv_tasks.size() + index_tasks.size());
      // For every one of our sub-operations, add an additional mapping
      // dependence.  When our sub-operations map, they will trigger these
      // mapping dependences which guarantees that we will not be able to
      // map until all of the sub-operations are ready to map.
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        indiv_tasks[idx]->execute_dependence_analysis();
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        index_tasks[idx]->execute_dependence_analysis();
    }

    //--------------------------------------------------------------------------
    /*static*/ bool MustEpochOp::single_task_sorter(
        const Task* t1, const Task* t2)
    //--------------------------------------------------------------------------
    {
      return (t1->index_point < t2->index_point);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_ready(void)
    //--------------------------------------------------------------------------
    {
      // Trigger al lthe individual and index tasks which will not actually
      // map and run them but instead enqueue them with our must epoch op
      // here so that we can call trigger mapping on ourselves once they
      // have all been enumerated
      legion_assert(!single_tasks_ready.exists());
      task_sets.resize(indiv_tasks.size() + index_tasks.size());
      // Add a guard on the single tasks being ready
      remaining_single_tasks.store(1);
      const Processor current = parent_ctx->get_executing_processor();
      for (IndividualTask* const & task : indiv_tasks)
      {
        task->prepare_map_must_epoch();
        task->set_target_proc(current);
        remaining_single_tasks.fetch_add(1);
        task->enqueue_ready_operation();
      }
      for (IndexTask* const & task : index_tasks)
      {
        task->prepare_map_must_epoch();
        task->set_target_proc(current);
        remaining_single_tasks.fetch_add(task->index_domain.get_volume());
        task->enqueue_ready_operation();
      }
      // Remove the guard that we added
      const unsigned remaining = remaining_single_tasks.fetch_sub(1);
      if (remaining > 1)
      {
        AutoLock o_lock(op_lock);
        // Make sure we didn't lose the race
        if (remaining_single_tasks.load() > 0)
          single_tasks_ready = Runtime::create_rt_user_event();
      }
      // Enqueue this as a ready operation once all the single tasks have
      // been enumerated for us to do the mapping
      enqueue_ready_operation(single_tasks_ready);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_mapping(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!single_tasks.empty());
      // Sort the points so that they are in order for determinism
      // across runs and for control replication
      std::sort(single_tasks.begin(), single_tasks.end(), single_task_sorter);
      // Then construct the inverse mapping
      for (unsigned idx = 0; idx < single_tasks.size(); idx++)
        single_task_map[single_tasks[idx]] = idx;
      // Next build the set of single tasks and all their constraints.
      // Iterate over all the recorded dependences
      std::vector<Mapper::MappingConstraint>& constraints = input.constraints;
      constraints.resize(dependences.size());
      mapping_dependences.resize(single_tasks.size());
      // Clear the dependence map now, we'll fill it in again
      // with a different set of points
      dependence_map.clear();
      unsigned constraint_idx = 0;
      for (DependenceRecord* it : dependences)
      {
        Mapper::MappingConstraint& constraint = constraints[constraint_idx];
        legion_assert(it->op_indexes.size() == it->req_indexes.size());
        // Add constraints for all the different elements
        std::set<unsigned> single_indexes;
        for (unsigned idx = 0; idx < it->op_indexes.size(); idx++)
        {
          unsigned req_index = it->req_indexes[idx];
          const std::set<SingleTask*>& task_set =
              task_sets[it->op_indexes[idx]];
          for (SingleTask* const & single_task : task_set)
          {
            constraint.constrained_tasks.emplace_back(single_task);
            constraint.requirement_indexes.emplace_back(req_index);
            legion_assert(
                single_task_map.find(single_task) != single_task_map.end());
            // Update the dependence map
            std::pair<unsigned, unsigned> key(
                single_task_map[single_task], req_index);
            dependence_map[key] = constraint_idx;
            single_indexes.insert(key.first);
          }
        }
        // Record the mapping dependences
        for (const unsigned& index1 : single_indexes)
        {
          for (const unsigned& index2 : single_indexes)
          {
            if (index2 == index1)
              break;
            mapping_dependences[index1].insert(index2);
          }
        }
        constraint_idx++;
      }
      // Clear this eagerly to save space
      for (DependenceRecord* record : dependences) delete record;
      dependences.clear();
      // Fill in the rest of the inputs to the mapper call
      input.mapping_tag = tag;
      input.tasks.insert(
          input.tasks.end(), single_tasks.begin(), single_tasks.end());
      // Also resize the outputs so the mapper knows what it is doing
      output.constraint_mappings.resize(input.constraints.size());
      output.task_processors.resize(single_tasks.size(), Processor::NO_PROC);
      // Now we can invoke the mapper
      MapperManager* mapper = invoke_mapper();
      // Check that all the tasks have been assigned to different processors
      // and perform the concurrent analysis on each of them so that they
      // all know what their starting preconditions are
      {
        std::map<Processor, SingleTask*> target_procs;
        for (unsigned idx = 0; idx < single_tasks.size(); idx++)
        {
          Processor proc = output.task_processors[idx];
          SingleTask* task = single_tasks[idx];
          if (!proc.exists())
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'map_must_epoch' "
                   "on mapper "
                << *mapper << ". Mapper failed to specify "
                << "a valid processor for " << *this << " at index " << idx
                << ".";
            error.raise();
          }
          if (target_procs.find(proc) != target_procs.end())
          {
            SingleTask* other = target_procs[proc];
            Error error(LEGION_MAPPER_EXCEPTION);
            error
                << "Invalid mapper output from invocation of 'map_must_epoch' "
                   "on mapper "
                << *mapper << ". Mapper requests both tasks " << *other
                << " and " << *task << " be mapped to the same "
                << "processor (" << proc
                << ") which is illegal in a must epoch launch.";
            error.raise();
          }
          target_procs[proc] = task;
          task->target_proc = proc;
        }
      }
      // Map and distribute all our tasks
      complete_mapping(map_and_distribute());
    }

    //--------------------------------------------------------------------------
    RtEvent MustEpochOp::map_and_distribute(void)
    //--------------------------------------------------------------------------
    {
      std::vector<RtEvent> tasks_all_mapped;
      tasks_all_mapped.reserve(indiv_tasks.size() + index_tasks.size());
      // Once all the tasks have been initialized we can defer
      // our all mapped event on all their all mapped events
      for (IndividualTask* const & task : indiv_tasks)
      {
        tasks_all_mapped.emplace_back(task->get_mapped_event());
        record_completion_effect(task->get_completion_event());
      }
      for (IndexTask* const & task : index_tasks)
      {
        tasks_all_mapped.emplace_back(task->get_mapped_event());
        record_completion_effect(task->get_completion_event());
      }
      // For correctness we still have to abide by the mapping dependences
      // computed on the individual tasks while we are mapping them
      for (SingleTask* const & task : single_tasks)
        mapped_events.emplace(
            std::make_pair(task->index_point, Runtime::create_rt_user_event()));
      remaining_mapped_events.store(single_tasks.size());
      remaining_collective_unbound_points = single_tasks.size();
      remaining_concurrent_mapped = single_tasks.size();
      remaining_concurrent_points = single_tasks.size();
      remaining_resource_returns = indiv_tasks.size() + slice_tasks.size();
      for (unsigned idx = 0; idx < single_tasks.size(); idx++)
      {
        // Figure out our preconditions
        std::vector<RtEvent> preconditions;
        for (const unsigned& it : mapping_dependences[idx])
        {
          legion_assert(it < idx);
          preconditions.emplace_back(
              mapped_events[single_tasks[it]->index_point]);
        }
        RtEvent precondition;
        if (!preconditions.empty())
          precondition = Runtime::merge_events(preconditions);
        if (precondition.exists() && !precondition.has_triggered())
          single_tasks[idx]->defer_perform_mapping(precondition, this);
        else if (
            single_tasks[idx]->perform_mapping(this) &&
            single_tasks[idx]->distribute_task())
          single_tasks[idx]->launch_task();
      }
      return Runtime::merge_events(tasks_all_mapped);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::record_mapped_event(
        const DomainPoint& point, RtEvent mapped)
    //--------------------------------------------------------------------------
    {
      legion_assert(mapped_events.find(point) != mapped_events.end());
      // No need for a lock since this data structure is read-only here
      Runtime::trigger_event(mapped_events[point], mapped);
      const unsigned remaining = remaining_mapped_events.fetch_sub(1);
      legion_assert(remaining > 0);
      if (remaining == 1)
      {
        std::vector<RtEvent> preconditions;
        preconditions.reserve(mapped_events.size());
        for (const std::pair<const DomainPoint, RtUserEvent>& it :
             mapped_events)
          preconditions.emplace_back(it.second);
        release_nonempty_acquired_instances(
            Runtime::merge_events(preconditions), acquired_instances);
      }
    }

    //--------------------------------------------------------------------------
    MapperManager* MustEpochOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
      Processor mapper_proc = parent_ctx->get_executing_processor();
      MapperManager* mapper = runtime->find_mapper(mapper_proc, map_id);
      // We've got all our meta-data set up so go ahead and issue the call
      mapper->invoke_map_must_epoch(this, input, output);
      return mapper;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::receive_resources(
        uint64_t return_index, std::map<LogicalRegion, unsigned>& created_regs,
        std::vector<DeletedRegion>& deleted_regs,
        std::set<std::pair<FieldSpace, FieldID> >& created_fids,
        std::vector<DeletedField>& deleted_fids,
        std::map<FieldSpace, unsigned>& created_fs,
        std::map<FieldSpace, std::set<LogicalRegion> >& latent_fs,
        std::vector<DeletedFieldSpace>& deleted_fs,
        std::map<IndexSpace, unsigned>& created_is,
        std::vector<DeletedIndexSpace>& deleted_is,
        std::map<IndexPartition, unsigned>& created_partitions,
        std::vector<DeletedPartition>& deleted_partitions,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      // Wait until we've received all the resources before handing them
      // back to the enclosing parent context
      {
        AutoLock o_lock(op_lock);
        merge_received_resources(
            created_regs, deleted_regs, created_fids, deleted_fids, created_fs,
            latent_fs, deleted_fs, created_is, deleted_is, created_partitions,
            deleted_partitions);
        legion_assert(remaining_resource_returns > 0);
        if (--remaining_resource_returns > 0)
          return;
      }
      // If we get here then we can finally do the return to the parent context
      // because we've received resources from all of our constituent operations
      return_resources(parent_ctx, context_index, preconditions);
    }

    //--------------------------------------------------------------------------
    uint64_t MustEpochOp::collective_lamport_allreduce(
        uint64_t lamport_clock, bool need_result)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      if (collective_lamport_clock < lamport_clock)
        collective_lamport_clock = lamport_clock;
      legion_assert(remaining_collective_unbound_points > 0);
      if (--remaining_collective_unbound_points > 0)
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
    void MustEpochOp::rendezvous_concurrent_mapped(RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      bool done;
      {
        AutoLock o_lock(op_lock);
        if (precondition.exists())
          concurrent_preconditions.emplace_back(precondition);
        legion_assert(remaining_concurrent_mapped > 0);
        done = (--remaining_concurrent_mapped == 0);
      }
      if (done)
        finalize_concurrent_mapped();
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::finalize_concurrent_mapped(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(concurrent_mapped.exists());
      Runtime::trigger_event(
          concurrent_mapped, Runtime::merge_events(concurrent_preconditions));
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::concurrent_allreduce(
        IndividualTask* task, AddressSpaceID space, uint64_t lamport_clock,
        bool poisoned)
    //--------------------------------------------------------------------------
    {
      bool done;
      {
        AutoLock o_lock(op_lock);
        if (poisoned)
          concurrent_poisoned = true;
        if (concurrent_lamport_clock < lamport_clock)
          concurrent_lamport_clock = lamport_clock;
        concurrent_tasks.emplace_back(std::make_pair(task, space));
        legion_assert(remaining_concurrent_points > 0);
        done = (--remaining_concurrent_points == 0);
      }
      if (done)
        finish_concurrent_allreduce();
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::concurrent_allreduce(
        SliceTask* slice, AddressSpaceID source, size_t total_points,
        uint64_t lamport_clock, bool poisoned)
    //--------------------------------------------------------------------------
    {
      bool done;
      {
        AutoLock o_lock(op_lock);
        if (poisoned)
          concurrent_poisoned = true;
        if (concurrent_lamport_clock < lamport_clock)
          concurrent_lamport_clock = lamport_clock;
        concurrent_slices.emplace_back(std::make_pair(slice, source));
        legion_assert(total_points <= remaining_concurrent_points);
        remaining_concurrent_points -= total_points;
        done = (remaining_concurrent_points == 0);
      }
      if (done)
        finish_concurrent_allreduce();
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::finish_concurrent_allreduce(void)
    //--------------------------------------------------------------------------
    {
      // Swap vectors onto the stack in case when we kick things off they
      // end up reclaiming the resources
      std::vector<std::pair<IndividualTask*, AddressSpaceID> > local_tasks;
      local_tasks.swap(concurrent_tasks);
      std::vector<std::pair<SliceTask*, AddressSpaceID> > local_slices;
      local_slices.swap(concurrent_slices);
      for (const std::pair<IndividualTask*, AddressSpaceID>& it : local_tasks)
      {
        if (it.second != runtime->address_space)
        {
          IndividualTaskConcurrentResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(it.first);
            rez.serialize(concurrent_lamport_clock);
            rez.serialize(concurrent_poisoned);
          }
          rez.dispatch(it.second);
        }
        else
          it.first->finish_concurrent_allreduce(
              concurrent_lamport_clock, concurrent_poisoned);
      }
      const Color color = 0;    // everything is color zero here
      const VariantID vid = 0;  // dummy variant since it's only for checking
      for (const std::pair<SliceTask*, AddressSpaceID>& it : local_slices)
      {
        if (it.second != runtime->address_space)
        {
          SliceConcurrentResponse rez;
          {
            RezCheck z(rez);
            rez.serialize(it.first);
            rez.serialize(color);
            rez.serialize(RtBarrier::NO_RT_BARRIER);
            rez.serialize(concurrent_lamport_clock);
            rez.serialize(vid);
            rez.serialize(concurrent_poisoned);
          }
          rez.dispatch(it.second);
        }
        else
          it.first->finish_concurrent_allreduce(
              color, concurrent_lamport_clock, concurrent_poisoned, vid,
              RtBarrier::NO_RT_BARRIER);
      }
    }

    //--------------------------------------------------------------------------
    IndexSpace MustEpochOp::compute_launch_space(
        const MustEpochLauncher& launcher, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      const size_t single_tasks = launcher.single_tasks.size();
      const size_t multi_tasks = launcher.index_tasks.size();
      legion_assert(!launch_domain.exists());
      legion_assert((single_tasks > 0) || (multi_tasks > 0));
      if (multi_tasks > 0)
      {
        if ((single_tasks > 0) || (multi_tasks > 1))
        {
          Realm::ProfilingRequestSet no_reqs;
          // Need to compute the index tasks
          switch (launcher.index_tasks[0].launch_domain.get_dim())
          {
#define DIMFUNC(DIM)                                                          \
  case DIM:                                                                   \
    {                                                                         \
      std::vector<Realm::IndexSpace<DIM, coord_t> > subspaces(                \
          single_tasks + multi_tasks);                                        \
      for (unsigned idx = 0; idx < multi_tasks; idx++)                        \
      {                                                                       \
        if (launcher.index_tasks[idx].launch_domain.exists())                 \
        {                                                                     \
          const Rect<DIM, coord_t> rect =                                     \
              launcher.index_tasks[idx].launch_domain;                        \
          subspaces[idx] = rect;                                              \
        }                                                                     \
        else                                                                  \
        {                                                                     \
          Domain domain;                                                      \
          runtime->find_domain(                                               \
              launcher.index_tasks[idx].launch_space, domain);                \
          const DomainT<DIM, coord_t> domaint = domain;                       \
          subspaces[idx] = domaint;                                           \
        }                                                                     \
      }                                                                       \
      for (unsigned idx = 0; idx < single_tasks; idx++)                       \
      {                                                                       \
        const Point<DIM, coord_t> p = launcher.single_tasks[idx].point;       \
        const Rect<DIM, coord_t> rect(p, p);                                  \
        subspaces[multi_tasks + idx] = Realm::IndexSpace<DIM, coord_t>(rect); \
      }                                                                       \
      Realm::IndexSpace<DIM, coord_t> space;                                  \
      const RtEvent wait_on(Realm::IndexSpace<DIM, coord_t>::compute_union(   \
          subspaces, space, no_reqs));                                        \
      const DomainT<DIM, coord_t> domaint(space);                             \
      launch_domain = domaint;                                                \
      if (wait_on.exists())                                                   \
        wait_on.wait();                                                       \
      break;                                                                  \
    }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              std::abort();
          }
          return parent_ctx->find_index_launch_space(
              launch_domain, provenance, true /*take ownership*/);
        }
        else  // Easy case of a single index task
        {
          launch_domain = launcher.index_tasks[0].launch_domain;
          if (!launch_domain.exists())
            runtime->find_domain(
                launcher.index_tasks[0].launch_space, launch_domain);
          return parent_ctx->find_index_launch_space(launch_domain, provenance);
        }
      }
      else
      {
        // These are just point tasks
        if (single_tasks > 1)
        {
          switch (launcher.single_tasks[0].point.get_dim())
          {
#define DIMFUNC(DIM)                                                        \
  case DIM:                                                                 \
    {                                                                       \
      std::vector<Realm::Point<DIM, coord_t> > points(single_tasks);        \
      for (unsigned idx = 0; idx < single_tasks; idx++)                     \
      {                                                                     \
        const Point<DIM, coord_t> point = launcher.single_tasks[idx].point; \
        points[idx] = point;                                                \
      }                                                                     \
      Realm::IndexSpace<DIM, coord_t> space(points);                        \
      const DomainT<DIM, coord_t> domaint(space);                           \
      launch_domain = domaint;                                              \
      break;                                                                \
    }
            LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
            default:
              std::abort();
          }
          return parent_ctx->find_index_launch_space(
              launch_domain, provenance, true /*take ownership*/);
        }
        else  // Easy case of a single point task
        {
          DomainPoint point = launcher.single_tasks[0].point;
          launch_domain = Domain(point, point);
          return parent_ctx->find_index_launch_space(launch_domain, provenance);
        }
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::trigger_commit(void)
    //--------------------------------------------------------------------------
    {
      bool need_commit;
      {
        AutoLock o_lock(op_lock);
        legion_assert(remaining_subop_commits > 0);
        remaining_subop_commits--;
        need_commit = (remaining_subop_commits == 0);
      }
      if (need_commit)
      {
        RtEvent commit_precondition;
        if (!commit_preconditions.empty())
          commit_precondition = Runtime::merge_events(commit_preconditions);
        commit_operation(true /*deactivate*/, commit_precondition);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::verify_dependence(
        Operation* src_op, GenerationID src_gen, Operation* dst_op,
        GenerationID dst_gen)
    //--------------------------------------------------------------------------
    {
      // If they are the same, then we can ignore them
      if ((src_op == dst_op) && (src_gen == dst_gen))
        return;
      // Check to see if the source is one of our operations, if it is
      // then we have an actual dependence which is an error.
      int src_index = find_operation_index(src_op, src_gen);
      if (src_index >= 0)
      {
        int dst_index = find_operation_index(dst_op, dst_gen);
        if (dst_index >= 0)
        {
          TaskOp* src_task = find_task_by_index(src_index);
          TaskOp* dst_task = find_task_by_index(dst_index);
          Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
          error
              << "Detected dependence between two tasks " << *src_task
              << " and " << *dst_task << ". Non-simulataneous dependences "
              << "between two tasks in a must-epoch launch are not permitted.";
          error.raise();
        }
      }
    }

    //--------------------------------------------------------------------------
    bool MustEpochOp::record_dependence(
        Operation* src_op, GenerationID src_gen, Operation* dst_op,
        GenerationID dst_gen, unsigned src_idx, unsigned dst_idx,
        DependenceType dtype)
    //--------------------------------------------------------------------------
    {
      // If they are the same we can ignore them
      if ((src_op == dst_op) && (src_gen == dst_gen))
        return true;
      // Check to see if the source is one of our operations
      int src_index = -1;
      int dst_index = find_operation_index(dst_op, dst_gen);
      if (dst_index < 0)
      {
        if (!internal_dependences.empty())
        {
          // Check to see if the destination is one of our internal ops
          std::pair<Operation*, GenerationID> internal_key(dst_op, dst_gen);
          std::map<
              std::pair<Operation*, GenerationID>,
              std::vector<std::pair<unsigned, unsigned> > >::const_iterator
              finder = internal_dependences.find(internal_key);
          if (finder != internal_dependences.end())
          {
            // should never have back-to-back internal ops
            legion_assert(!src_op->is_internal_op());
            src_index = find_operation_index(src_op, src_gen);
            legion_assert(src_index >= 0);
            TaskOp* src_task = find_task_by_index(src_index);
            const RegionRequirement& src_req = src_task->regions[src_idx];
            IndexSpaceNode* src_node =
                runtime->get_node(src_req.region.get_index_space());
            // Scan through all the dependences this internal operation
            // had on other tasks inside the must epoch launch and see
            // which ones we actually interfere with so we can record
            // the appropriate constraints
            for (const std::pair<unsigned, unsigned>& it : finder->second)
            {
              TaskOp* dst_task = find_task_by_index(it.first);
              const RegionRequirement& dst_req = dst_task->regions[it.second];
              IndexSpaceNode* dst_node =
                  runtime->get_node(dst_req.region.get_index_space());
              if (runtime->are_disjoint_tree_only(src_node, dst_node))
                continue;
              if (!src_node->intersects_with(dst_node))
                continue;
              // Update the dependence type
              DependenceType internal_dtype =
                  check_dependence_type<true, true /*reductions interfere*/>(
                      RegionUsage(src_req), RegionUsage(dst_req));
              record_intra_must_epoch_dependence(
                  src_index, src_idx, it.first, it.second, internal_dtype);
            }
          }
        }
        return true;
      }
      if (src_op->is_internal_op())
      {
        // Refinement operations should not record dependences on previous
        // operations in the same must epoch operation
        legion_assert(src_op->get_operation_kind() == REFINEMENT_OP_KIND);
        InternalOp* internal_op = legion_safe_cast<InternalOp*>(src_op);
        // Record the destination as a potential target for anything
        // that comes later and depends on the internal operation
        std::pair<Operation*, GenerationID> internal_key(src_op, src_gen);
        internal_dependences[internal_key].emplace_back(
            std::pair<unsigned, unsigned>(dst_index, dst_idx));
        // Use the source of the internal operation here since we still
        // need to record constraints properly between these operations
        src_index = find_operation_index(
            internal_op->get_creator_op(), internal_op->get_creator_gen());
        legion_assert(src_index >= 0);  // better be able to find it
        src_idx = internal_op->get_internal_index();
        TaskOp* src_task = find_task_by_index(src_index);
        TaskOp* dst_task = find_task_by_index(dst_index);
        const RegionRequirement& src_req = src_task->regions[src_idx];
        const RegionRequirement& dst_req = dst_task->regions[dst_idx];
        legion_assert(src_req.handle_type == LEGION_SINGULAR_PROJECTION);
        legion_assert(dst_req.handle_type == LEGION_SINGULAR_PROJECTION);
        // Check to see if the regions actually do interfere
        IndexSpaceNode* src_node =
            runtime->get_node(src_req.region.get_index_space());
        IndexSpaceNode* dst_node =
            runtime->get_node(dst_req.region.get_index_space());
        if (runtime->are_disjoint_tree_only(src_node, dst_node))
          return false;
        if (!src_node->intersects_with(dst_node))
          return false;
        // Update the dependence type
        dtype = check_dependence_type<true, true /*reductions interfere*/>(
            RegionUsage(src_req), RegionUsage(dst_req));
      }
      else
        src_index = find_operation_index(src_op, src_gen);
      legion_assert(src_index >= 0);
      return record_intra_must_epoch_dependence(
          src_index, src_idx, dst_index, dst_idx, dtype);
    }

    //--------------------------------------------------------------------------
    bool MustEpochOp::record_intra_must_epoch_dependence(
        unsigned src_index, unsigned src_idx, unsigned dst_index,
        unsigned dst_idx, DependenceType dtype)
    //--------------------------------------------------------------------------
    {
      // If it is, see what kind of dependence we have
      if ((dtype == LEGION_TRUE_DEPENDENCE) ||
          (dtype == LEGION_ANTI_DEPENDENCE) ||
          (dtype == LEGION_ATOMIC_DEPENDENCE))
      {
        TaskOp* src_task = find_task_by_index(src_index);
        TaskOp* dst_task = find_task_by_index(dst_index);
        Error error(LEGION_PROGRAMMING_MODEL_EXCEPTION);
        error << "Detected dependence between region " << src_idx << " of "
              << *src_task << " and " << dst_idx << " of " << *dst_task
              << " of type "
              << ((dtype == LEGION_TRUE_DEPENDENCE) ? "TRUE DEPENDENCE" :
                  (dtype == LEGION_ANTI_DEPENDENCE) ? "ANTI DEPENDENCE" :
                                                      "ATOMIC DEPENDENCE")
              << ". Non-simultaneous dependences between two tasks in a "
              << "must epoch_launch are not permitted.";
        error.raise();
      }
      else if (dtype == LEGION_SIMULTANEOUS_DEPENDENCE)
      {
        // See if the dependence record already exists
        const std::pair<unsigned, unsigned> src_key(src_index, src_idx);
        const std::pair<unsigned, unsigned> dst_key(dst_index, dst_idx);
        std::map<std::pair<unsigned, unsigned>, unsigned>::iterator
            src_record_finder = dependence_map.find(src_key);
        if (src_record_finder != dependence_map.end())
        {
          // Already have a source record, see if we have
          // a destination record too
          std::map<std::pair<unsigned, unsigned>, unsigned>::iterator
              dst_record_finder = dependence_map.find(dst_key);
          if (dst_record_finder == dependence_map.end())
          {
            // Update the destination record entry
            dependence_map[dst_key] = src_record_finder->second;
            dependences[src_record_finder->second]->add_entry(
                dst_index, dst_idx);
          }
          else  // both already there so just assert they are the same
            legion_assert(
                src_record_finder->second == dst_record_finder->second);
        }
        else
        {
          // No source record
          // See if we have a destination record entry
          std::map<std::pair<unsigned, unsigned>, unsigned>::iterator
              dst_record_finder = dependence_map.find(dst_key);
          if (dst_record_finder == dependence_map.end())
          {
            // Neither source nor destination have an entry so
            // make a new record
            DependenceRecord* new_record = new DependenceRecord();
            new_record->add_entry(src_index, src_idx);
            new_record->add_entry(dst_index, dst_idx);
            unsigned record_index = dependences.size();
            dependence_map[src_key] = record_index;
            dependence_map[dst_key] = record_index;
            dependences.emplace_back(new_record);
          }
          else
          {
            // Have a destination but no source, so update the source
            dependence_map[src_key] = dst_record_finder->second;
            dependences[dst_record_finder->second]->add_entry(
                src_index, src_idx);
          }
        }
        return false;
      }
      // NO_DEPENDENCE and PROMOTED_DEPENDENCE are not errors
      // and do not need to be recorded
      return true;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::must_epoch_map_task_callback(
        SingleTask* task, Mapper::MapTaskInput& map_input,
        Mapper::MapTaskOutput& map_output)
    //--------------------------------------------------------------------------
    {
      // We have to do three things here
      // 1. Update the target processor
      // 2. Mark as inputs and outputs any regions which we know
      //    the results for as a result of our must epoch mapping
      // 3. Record that we premapped those regions
      // First find the index for this task
      legion_assert(single_task_map.find(task) != single_task_map.end());
      unsigned index = single_task_map[task];
      // Set the target processor by the index
      task->target_proc = output.task_processors[index];
      // Now iterate over the constraints figure out which ones
      // apply to this task
      std::pair<unsigned, unsigned> key(index, 0);
      for (unsigned idx = 0; idx < task->regions.size(); idx++)
      {
        key.second = idx;
        std::map<std::pair<unsigned, unsigned>, unsigned>::const_iterator
            record_finder = dependence_map.find(key);
        if (record_finder != dependence_map.end())
        {
          map_input.valid_instances[idx] =
              output.constraint_mappings[record_finder->second];
          map_output.chosen_instances[idx] =
              output.constraint_mappings[record_finder->second];
          // Also record that we premapped this
          map_input.premapped_regions.emplace_back(idx);
        }
      }
    }

    //--------------------------------------------------------------------------
    std::map<PhysicalManager*, unsigned>*
        MustEpochOp::get_acquired_instances_ref(void)
    //--------------------------------------------------------------------------
    {
      return &acquired_instances;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::register_single_task(SingleTask* single, unsigned index)
    //--------------------------------------------------------------------------
    {
      // Can do the first part without the lock
      legion_assert(index < task_sets.size());
      task_sets[index].insert(single);
      AutoLock o_lock(op_lock);
      single_tasks.emplace_back(single);
      const unsigned remaining = remaining_single_tasks.fetch_sub(1);
      legion_assert(remaining > 0);
      if ((remaining == 1) && single_tasks_ready.exists())
        Runtime::trigger_event(single_tasks_ready);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::register_slice_task(SliceTask* slice)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      slice_tasks.insert(slice);
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::register_subop(Operation* op)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      remaining_subop_completes++;
      remaining_subop_commits++;
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::notify_subop_complete(Operation* op, ApEvent effect)
    //--------------------------------------------------------------------------
    {
      if (effect.exists())
        record_completion_effect(effect);
      bool need_complete;
      {
        AutoLock o_lock(op_lock);
        legion_assert(remaining_subop_completes > 0);
        remaining_subop_completes--;
        need_complete = (remaining_subop_completes == 0);
      }
      if (need_complete)
        complete_execution();
    }

    //--------------------------------------------------------------------------
    void MustEpochOp::notify_subop_commit(Operation* op, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      bool need_commit;
      {
        AutoLock o_lock(op_lock);
        legion_assert(remaining_subop_commits > 0);
        if (precondition.exists())
          commit_preconditions.insert(precondition);
        remaining_subop_commits--;
        need_commit = (remaining_subop_commits == 0);
      }
      if (need_commit)
      {
        RtEvent commit_precondition;
        if (!commit_preconditions.empty())
          commit_precondition = Runtime::merge_events(commit_preconditions);
        commit_operation(true /*deactivate*/, commit_precondition);
      }
    }

    //--------------------------------------------------------------------------
    RtUserEvent MustEpochOp::find_slice_versioning_event(
        UniqueID slice_id, bool& first)
    //--------------------------------------------------------------------------
    {
      AutoLock o_lock(op_lock);
      std::map<UniqueID, RtUserEvent>::const_iterator finder =
          slice_version_events.find(slice_id);
      if (finder == slice_version_events.end())
      {
        first = true;
        RtUserEvent result = Runtime::create_rt_user_event();
        slice_version_events[slice_id] = result;
        return result;
      }
      else
      {
        first = false;
        return finder->second;
      }
    }

    //--------------------------------------------------------------------------
    int MustEpochOp::find_operation_index(Operation* op, GenerationID op_gen)
    //--------------------------------------------------------------------------
    {
      if (op->get_operation_kind() != TASK_OP_KIND)
        return -1;
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        if ((indiv_tasks[idx] == op) &&
            (indiv_tasks[idx]->get_generation() == op_gen))
          return idx;
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        if ((index_tasks[idx] == op) &&
            (index_tasks[idx]->get_generation() == op_gen))
          return (idx + indiv_tasks.size());
      }
      return -1;
    }

    //--------------------------------------------------------------------------
    TaskOp* MustEpochOp::find_task_by_index(int index)
    //--------------------------------------------------------------------------
    {
      legion_assert(index >= 0);
      if ((size_t)index < indiv_tasks.size())
        return indiv_tasks[index];
      index -= indiv_tasks.size();
      if ((size_t)index < index_tasks.size())
        return index_tasks[index];
      std::abort();
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Mapping Broadcast
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochMappingBroadcast::MustEpochMappingBroadcast(
        ReplicateContext* ctx, ShardID origin, CollectiveID collective_id)
      : BroadcastCollective(ctx, collective_id, origin)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MustEpochMappingBroadcast::~MustEpochMappingBroadcast(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(local_done_event.exists());
      if (!done_events.empty())
        Runtime::trigger_event(
            local_done_event, Runtime::merge_events(done_events));
      else
        Runtime::trigger_event(local_done_event);
      // This should only happen on the owner node
      if (!held_references.empty())
      {
        // Wait for all the other shards to be done
        local_done_event.wait();
        // Now we can remove our held references
        for (PhysicalManager* manager : held_references)
          if (manager->remove_base_valid_ref(REPLICATION_REF))
            delete manager;
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingBroadcast::pack_collective(Serializer& rez) const
    //--------------------------------------------------------------------------
    {
      RtUserEvent next_done = Runtime::create_rt_user_event();
      done_events.insert(next_done);
      rez.serialize(next_done);
      rez.serialize<size_t>(processors.size());
      for (unsigned idx = 0; idx < processors.size(); idx++)
        rez.serialize(processors[idx]);
      rez.serialize<size_t>(instances.size());
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const std::vector<DistributedID>& dids = instances[idx];
        rez.serialize<size_t>(dids.size());
        for (const DistributedID& it : dids) rez.serialize(it);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingBroadcast::unpack_collective(Deserializer& derez)
    //--------------------------------------------------------------------------
    {
      derez.deserialize(local_done_event);
      size_t num_procs;
      derez.deserialize(num_procs);
      processors.resize(num_procs);
      for (unsigned idx = 0; idx < num_procs; idx++)
        derez.deserialize(processors[idx]);
      size_t num_constraints;
      derez.deserialize(num_constraints);
      instances.resize(num_constraints);
      for (unsigned idx1 = 0; idx1 < num_constraints; idx1++)
      {
        size_t num_dids;
        derez.deserialize(num_dids);
        std::vector<DistributedID>& dids = instances[idx1];
        dids.resize(num_dids);
        for (unsigned idx2 = 0; idx2 < num_dids; idx2++)
          derez.deserialize(dids[idx2]);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingBroadcast::broadcast(
        const std::vector<Processor>& processor_mapping,
        const std::vector<std::vector<Mapping::PhysicalInstance> >& mappings)
    //--------------------------------------------------------------------------
    {
      legion_assert(!local_done_event.exists());
      local_done_event = Runtime::create_rt_user_event();
      processors = processor_mapping;
      instances.resize(mappings.size());
      // Add valid references to all the physical instances that we will
      // hold until all the must epoch operations are done with the exchange
      for (unsigned idx1 = 0; idx1 < mappings.size(); idx1++)
      {
        std::vector<DistributedID>& dids = instances[idx1];
        dids.resize(mappings[idx1].size());
        for (unsigned idx2 = 0; idx2 < dids.size(); idx2++)
        {
          const Mapping::PhysicalInstance& inst = mappings[idx1][idx2];
          PhysicalManager* manager = inst.impl->as_physical_manager();
          dids[idx2] = manager->did;
          if (held_references.find(manager) != held_references.end())
            continue;
          manager->add_base_valid_ref(REPLICATION_REF);
          held_references.insert(manager);
        }
      }
      perform_collective_async();
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingBroadcast::receive_results(
        std::vector<Processor>& processor_mapping,
        const std::vector<unsigned>& constraint_indexes,
        std::vector<std::vector<Mapping::PhysicalInstance> >& mappings,
        std::map<PhysicalManager*, unsigned>& acquired)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      // Just grab all the processors since we still need them
      processor_mapping = processors;
      // We are a little smarter with the mappings since we know exactly
      // which ones we are actually going to need for our local points
      std::set<RtEvent> ready_events;
      for (const unsigned& index : constraint_indexes)
      {
        legion_assert(index < instances.size());
        legion_assert(index < mappings.size());
        const std::vector<DistributedID>& dids = instances[index];
        std::vector<Mapping::PhysicalInstance>& mapping = mappings[index];
        mapping.resize(dids.size());
        for (unsigned idx = 0; idx < dids.size(); idx++)
        {
          RtEvent ready;
          mapping[idx].impl =
              runtime->find_or_request_instance_manager(dids[idx], ready);
          if (!ready.has_triggered())
            ready_events.insert(ready);
        }
      }
      // Have to wait for the ready events to trigger before we can add
      // our references safely
      if (!ready_events.empty())
      {
        RtEvent ready = Runtime::merge_events(ready_events);
        if (!ready.has_triggered())
          ready.wait();
      }
      // Lastly we need to put acquire references on any of local instances
      for (unsigned idx = 0; idx < constraint_indexes.size(); idx++)
      {
        const unsigned constraint_index = constraint_indexes[idx];
        const std::vector<Mapping::PhysicalInstance>& mapping =
            mappings[constraint_index];
        // Also grab an acquired reference to these instances
        for (const Mapping::PhysicalInstance& it : mapping)
        {
          PhysicalManager* manager = it.impl->as_physical_manager();
          // If we already had a reference to this instance
          // then we don't need to add any additional ones
          if (acquired.find(manager) != acquired.end())
            continue;
          manager->add_base_gc_ref(MAPPER_REF);
          legion_no_skip_assert(manager->acquire_instance(MAPPING_ACQUIRE_REF));
          acquired[manager] = 1 /*count*/;
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Mapping Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochMappingExchange::MustEpochMappingExchange(
        ReplicateContext* ctx, CollectiveID collective_id)
      : AllGatherCollective(ctx, collective_id)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MustEpochMappingExchange::~MustEpochMappingExchange(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(local_done_event.exists());  // better have one of these
      Runtime::trigger_event(local_done_event);
      // See if we need to wait for others to be done before we can
      // remove our valid references
      if (!done_events.empty())
      {
        RtEvent done = Runtime::merge_events(done_events);
        if (!done.has_triggered())
          done.wait();
      }
      // Now we can remove our held references
      for (PhysicalManager* manager : held_references)
        if (manager->remove_base_valid_ref(REPLICATION_REF))
          delete manager;
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(processors.size());
      for (const std::pair<const DomainPoint, Processor>& it : processors)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
      rez.serialize<size_t>(constraints.size());
      for (const std::pair<const unsigned, ConstraintInfo>& it : constraints)
      {
        rez.serialize(it.first);
        rez.serialize<size_t>(it.second.instances.size());
        for (unsigned idx = 0; idx < it.second.instances.size(); idx++)
          rez.serialize(it.second.instances[idx]);
        rez.serialize(it.second.origin_shard);
        rez.serialize(it.second.weight);
      }
      rez.serialize<size_t>(done_events.size());
      for (const RtEvent& done_event : done_events) rez.serialize(done_event);
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_procs;
      derez.deserialize(num_procs);
      for (unsigned idx = 0; idx < num_procs; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        derez.deserialize(processors[point]);
      }
      size_t num_mappings;
      derez.deserialize(num_mappings);
      for (unsigned idx1 = 0; idx1 < num_mappings; idx1++)
      {
        unsigned constraint_index;
        derez.deserialize(constraint_index);
        std::map<unsigned, ConstraintInfo>::iterator finder =
            constraints.find(constraint_index);
        if (finder == constraints.end())
        {
          // Can unpack directly since we're first
          ConstraintInfo& info = constraints[constraint_index];
          size_t num_dids;
          derez.deserialize(num_dids);
          info.instances.resize(num_dids);
          for (unsigned idx2 = 0; idx2 < num_dids; idx2++)
            derez.deserialize(info.instances[idx2]);
          derez.deserialize(info.origin_shard);
          derez.deserialize(info.weight);
        }
        else
        {
          // Unpack into a temporary
          ConstraintInfo info;
          size_t num_dids;
          derez.deserialize(num_dids);
          info.instances.resize(num_dids);
          for (unsigned idx2 = 0; idx2 < num_dids; idx2++)
            derez.deserialize(info.instances[idx2]);
          derez.deserialize(info.origin_shard);
          derez.deserialize(info.weight);
          // Only keep the result if we have a larger weight
          // or we have the same weight and a smaller shard
          if ((info.weight > finder->second.weight) ||
              ((info.weight == finder->second.weight) &&
               (info.origin_shard < finder->second.origin_shard)))
            finder->second = info;
        }
      }
      size_t num_done;
      derez.deserialize(num_done);
      for (unsigned idx = 0; idx < num_done; idx++)
      {
        RtEvent done_event;
        derez.deserialize(done_event);
        done_events.insert(done_event);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochMappingExchange::exchange_must_epoch_mappings(
        ShardID shard_id, size_t total_shards, size_t total_constraints,
        const std::vector<const Task*>& local_tasks,
        const std::vector<const Task*>& all_tasks,
        std::vector<Processor>& processor_mapping,
        const std::vector<unsigned>& constraint_indexes,
        std::vector<std::vector<Mapping::PhysicalInstance> >& mappings,
        const std::vector<int>& mapping_weights,
        std::map<PhysicalManager*, unsigned>& acquired)
    //--------------------------------------------------------------------------
    {
      legion_assert(local_tasks.size() == processor_mapping.size());
      legion_assert(constraint_indexes.size() == mappings.size());
      // Add valid references to all the physical instances that we will
      // hold until all the must epoch operations are done with the exchange
      for (unsigned idx = 0; idx < mappings.size(); idx++)
      {
        for (const Mapping::PhysicalInstance& it : mappings[idx])
        {
          PhysicalManager* manager = it.impl->as_physical_manager();
          if (held_references.find(manager) != held_references.end())
            continue;
          manager->add_base_valid_ref(REPLICATION_REF);
          held_references.insert(manager);
        }
      }
      legion_assert(!local_done_event.exists());
      local_done_event = Runtime::create_rt_user_event();
      // Then we can add our instances to the set and do the exchange
      {
        AutoLock c_lock(collective_lock);
        for (unsigned idx = 0; idx < local_tasks.size(); idx++)
        {
          const Task* task = local_tasks[idx];
          legion_assert(processors.find(task->index_point) == processors.end());
          processors[task->index_point] = processor_mapping[idx];
        }
        for (unsigned idx1 = 0; idx1 < mappings.size(); idx1++)
        {
          const unsigned constraint_index = constraint_indexes[idx1];
          legion_assert(constraint_index < total_constraints);
          std::map<unsigned, ConstraintInfo>::iterator finder =
              constraints.find(constraint_index);
          // Only add it if it doesn't exist or it has a lower weight
          // or it has the same weight and is a lower shard
          if ((finder == constraints.end()) ||
              (mapping_weights[idx1] > finder->second.weight) ||
              ((mapping_weights[idx1] == finder->second.weight) &&
               (shard_id < finder->second.origin_shard)))
          {
            ConstraintInfo& info = constraints[constraint_index];
            info.instances.resize(mappings[idx1].size());
            for (unsigned idx2 = 0; idx2 < mappings[idx1].size(); idx2++)
              info.instances[idx2] = mappings[idx1][idx2].impl->did;
            info.origin_shard = shard_id;
            info.weight = mapping_weights[idx1];
          }
        }
        // Also update the local done events
        done_events.insert(local_done_event);
      }
      perform_collective_sync();
      // Start fetching the all the mapping results to get them in flight
      mappings.clear();
      mappings.resize(total_constraints);
      std::set<RtEvent> ready_events;
      // We only need to get the results for local constraints as we
      // know that we aren't going to care about any of the rest
      for (unsigned idx1 = 0; idx1 < constraint_indexes.size(); idx1++)
      {
        const unsigned constraint_index = constraint_indexes[idx1];
        const std::vector<DistributedID>& dids =
            constraints[constraint_index].instances;
        std::vector<Mapping::PhysicalInstance>& mapping =
            mappings[constraint_index];
        mapping.resize(dids.size());
        for (unsigned idx2 = 0; idx2 < dids.size(); idx2++)
        {
          RtEvent ready;
          mapping[idx2].impl =
              runtime->find_or_request_instance_manager(dids[idx2], ready);
          if (!ready.has_triggered())
            ready_events.insert(ready);
        }
      }
      // Update the processor mapping
      processor_mapping.resize(all_tasks.size());
      for (unsigned idx = 0; idx < all_tasks.size(); idx++)
      {
        const Task* task = all_tasks[idx];
        std::map<DomainPoint, Processor>::const_iterator finder =
            processors.find(task->index_point);
        legion_assert(finder != processors.end());
        processor_mapping[idx] = finder->second;
      }
      // Wait for all the instances to be ready
      if (!ready_events.empty())
      {
        RtEvent ready = Runtime::merge_events(ready_events);
        if (!ready.has_triggered())
          ready.wait();
      }
      // Lastly we need to put acquire references on any of local instances
      for (unsigned idx = 0; idx < constraint_indexes.size(); idx++)
      {
        const unsigned constraint_index = constraint_indexes[idx];
        const std::vector<Mapping::PhysicalInstance>& mapping =
            mappings[constraint_index];
        // Also grab an acquired reference to these instances
        for (const Mapping::PhysicalInstance& it : mapping)
        {
          PhysicalManager* manager = it.impl->as_physical_manager();
          // If we already had a reference to this instance
          // then we don't need to add any additional ones
          if (acquired.find(manager) != acquired.end())
            continue;
          manager->add_base_gc_ref(MAPPER_REF);
          manager->add_base_valid_ref(MAPPING_ACQUIRE_REF);
          acquired[manager] = 1 /*count*/;
        }
      }
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Dependence Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochDependenceExchange::MustEpochDependenceExchange(
        CollectiveID id, ReplicateContext* ctx,
        std::map<DomainPoint, RtUserEvent>& events)
      : AllGatherCollective(ctx, id), mapped_events(events)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MustEpochDependenceExchange::~MustEpochDependenceExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void MustEpochDependenceExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      rez.serialize<size_t>(mapped_events.size());
      for (const std::pair<const DomainPoint, RtUserEvent>& it : mapped_events)
      {
        rez.serialize(it.first);
        rez.serialize(it.second);
      }
    }

    //--------------------------------------------------------------------------
    void MustEpochDependenceExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      size_t num_deps;
      derez.deserialize(num_deps);
      for (unsigned idx = 0; idx < num_deps; idx++)
      {
        DomainPoint point;
        derez.deserialize(point);
        derez.deserialize(mapped_events[point]);
      }
    }

    /////////////////////////////////////////////////////////////
    // Must Epoch Completion Exchange
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MustEpochCompletionExchange::MustEpochCompletionExchange(
        CollectiveID id, ReplicateContext* ctx,
        std::vector<RtEvent>& local_mapped,
        std::vector<ApEvent>& local_complete)
      : AllGatherCollective(ctx, id), local_mapped_events(local_mapped),
        local_complete_events(local_complete)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    MustEpochCompletionExchange::~MustEpochCompletionExchange(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void MustEpochCompletionExchange::pack_collective_stage(
        ShardID target, Serializer& rez, int stage)
    //--------------------------------------------------------------------------
    {
      if (local_mapped_events.size() > 1)
      {
        RtEvent next_local = Runtime::merge_events(local_mapped_events);
        local_mapped_events.clear();
        if (next_local.exists())
          local_mapped_events.emplace_back(next_local);
      }
      if (local_complete_events.size() > 1)
      {
        ApEvent next_local =
            Runtime::merge_events(nullptr, local_complete_events);
        local_complete_events.clear();
        if (next_local.exists())
          local_complete_events.emplace_back(next_local);
      }
      if (local_mapped_events.empty())
        rez.serialize(RtEvent::NO_RT_EVENT);
      else
        rez.serialize(local_mapped_events.back());
      if (local_complete_events.empty())
        rez.serialize(ApEvent::NO_AP_EVENT);
      else
        rez.serialize(local_complete_events.back());
    }

    //--------------------------------------------------------------------------
    void MustEpochCompletionExchange::unpack_collective_stage(
        Deserializer& derez, int stage)
    //--------------------------------------------------------------------------
    {
      RtEvent remote_mapped;
      derez.deserialize(remote_mapped);
      if (remote_mapped.exists())
        local_mapped_events.emplace_back(remote_mapped);
      ApEvent remote_complete;
      derez.deserialize(remote_complete);
      if (!remote_complete.exists())
        local_complete_events.emplace_back(remote_complete);
    }

    //--------------------------------------------------------------------------
    RtEvent MustEpochCompletionExchange::finish_exchange(ReplMustEpochOp* op)
    //--------------------------------------------------------------------------
    {
      perform_collective_wait();
      if (!local_complete_events.empty())
        op->record_completion_effects(local_complete_events);
      return Runtime::merge_events(local_mapped_events);
    }

    /////////////////////////////////////////////////////////////
    // Repl Must Epoch Op
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ReplMustEpochOp::ReplMustEpochOp(void) : MustEpochOp()
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    ReplMustEpochOp::~ReplMustEpochOp(void)
    //--------------------------------------------------------------------------
    { }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::activate(void)
    //--------------------------------------------------------------------------
    {
      MustEpochOp::activate();
      sharding_functor = std::numeric_limits<ShardingID>::max();
      sharding_function = nullptr;
      mapping_collective_id = 0;
      collective_map_must_epoch_call = false;
      mapping_broadcast = nullptr;
      mapping_exchange = nullptr;
      concurrent_exchange = nullptr;
      collective_exchange = nullptr;
      collective_exchange_id = 0;
      dependence_exchange_id = 0;
      completion_exchange_id = 0;
      concurrent_mapped_barrier = RtBarrier::NO_RT_BARRIER;
      resource_return_barrier = RtBarrier::NO_RT_BARRIER;
      sharding_collective = nullptr;
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::deactivate(bool freeop)
    //--------------------------------------------------------------------------
    {
      shard_single_tasks.clear();
      if (mapping_broadcast != nullptr)
        delete mapping_broadcast;
      if (mapping_exchange != nullptr)
        delete mapping_exchange;
      if (collective_exchange != nullptr)
        delete collective_exchange;
      if (concurrent_exchange != nullptr)
        delete concurrent_exchange;
      if (sharding_collective != nullptr)
        delete sharding_collective;
      MustEpochOp::deactivate(false /*free*/);
      if (freeop)
        runtime->free_operation(this);
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::instantiate_tasks(
        InnerContext* ctx, const MustEpochLauncher& launcher)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx = legion_safe_cast<ReplicateContext*>(ctx);
      legion_assert(!concurrent_mapped.exists());
      legion_assert(!concurrent_mapped_barrier.exists());
      // This is a bit dumb, but we do it to make the types work out
      concurrent_mapped = Runtime::create_rt_user_event();
      concurrent_mapped_barrier =
          repl_ctx->get_next_must_epoch_mapped_barrier();
      Runtime::trigger_event(concurrent_mapped, concurrent_mapped_barrier);
      Provenance* provenance = get_provenance();
      // Initialize operations for everything in the launcher
      // Note that we do not track these operations as we want them all to
      // appear as a single operation to the parent context in order to
      // avoid deadlock with the maximum window size.
      indiv_tasks.resize(launcher.single_tasks.size());
      for (unsigned idx = 0; idx < launcher.single_tasks.size(); idx++)
      {
        ReplIndividualTask* task = runtime->get_operation<ReplIndividualTask>();
        task->initialize_task(
            ctx, launcher.single_tasks[idx], provenance, false /*top level*/,
            true /*must epoch*/);
        task->set_must_epoch(this, idx, true /*register*/);
        // If we have a trace, set it for this operation as well
        if (trace != nullptr)
          trace->initialize_operation(task, nullptr);
        task->must_epoch_task = true;
        task->initialize_replication(repl_ctx);
        task->index_domain = this->launch_domain;
        task->sharding_space = launcher.sharding_space;
        if (runtime->safe_mapper)
          task->set_sharding_collective(new ShardingGatherCollective(
              repl_ctx, 0 /*owner shard*/, COLLECTIVE_LOC_59));
        indiv_tasks[idx] = task;
        indiv_tasks[idx]->set_concurrent_postcondition(concurrent_mapped);
      }
      index_tasks.resize(launcher.index_tasks.size());
      for (unsigned idx = 0; idx < launcher.index_tasks.size(); idx++)
      {
        IndexSpace launch_space = launcher.index_tasks[idx].launch_space;
        if (!launch_space.exists())
          launch_space = ctx->find_index_launch_space(
              launcher.index_tasks[idx].launch_domain, provenance);
        ReplIndexTask* task = runtime->get_operation<ReplIndexTask>();
        task->initialize_task(
            ctx, launcher.index_tasks[idx], launch_space, provenance,
            false /*track*/);
        task->set_must_epoch(this, indiv_tasks.size() + idx, true /*register*/);
        if (trace != nullptr)
          trace->initialize_operation(task, nullptr);
        task->must_epoch_task = true;
        task->initialize_replication(repl_ctx);
        task->sharding_space = launcher.sharding_space;
        if (runtime->safe_mapper)
          task->set_sharding_collective(new ShardingGatherCollective(
              repl_ctx, 0 /*owner shard*/, COLLECTIVE_LOC_59));
        index_tasks[idx] = task;
        index_tasks[idx]->initialize_must_epoch_concurrent_group(
            0 /*color*/, concurrent_mapped);
      }
    }

    //--------------------------------------------------------------------------
    FutureMap ReplMustEpochOp::create_future_map(
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
      return repl_ctx->shard_manager->deduplicate_future_map_creation(
          repl_ctx, this, launch_node, shard_node, future_map_did,
          get_provenance());
    }

    //--------------------------------------------------------------------------
    MapperManager* ReplMustEpochOp::invoke_mapper(void)
    //--------------------------------------------------------------------------
    {
      Processor mapper_proc = parent_ctx->get_executing_processor();
      MapperManager* mapper = runtime->find_mapper(mapper_proc, map_id);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // We want to do the map must epoch call
      // First find all the tasks that we own on this shard
      Domain shard_domain = launch_domain;
      if (sharding_space.exists())
        runtime->find_domain(sharding_space, shard_domain);
      for (SingleTask* task : single_tasks)
      {
        const ShardID shard =
            sharding_function->find_owner(task->index_point, shard_domain);
        LegionSpy::log_owner_shard(task->get_unique_id(), shard);
        // If it is not our shard then we don't own it
        if (shard != repl_ctx->owner_shard->shard_id)
          continue;
        shard_single_tasks.insert(task);
      }
      // Find the set of constraints that apply to our local set of tasks
      std::vector<Mapper::MappingConstraint> local_constraints;
      std::vector<unsigned> original_constraint_indexes;
      for (unsigned idx = 0; idx < input.constraints.size(); idx++)
      {
        bool is_local = false;
        for (const Task* task : input.constraints[idx].constrained_tasks)
        {
          SingleTask* single =
              static_cast<SingleTask*>(const_cast<Task*>(task));
          if (shard_single_tasks.find(single) == shard_single_tasks.end())
            continue;
          is_local = true;
          break;
        }
        if (is_local)
        {
          local_constraints.emplace_back(input.constraints[idx]);
          original_constraint_indexes.emplace_back(idx);
        }
      }
      if (collective_map_must_epoch_call)
      {
        // Update the input tasks for our subset
        std::vector<const Task*> all_tasks(
            shard_single_tasks.begin(), shard_single_tasks.end());
        input.tasks.swap(all_tasks);
        // Sort them again by their index points to for determinism
        std::sort(input.tasks.begin(), input.tasks.end(), single_task_sorter);
        // Update the constraints to contain just our subset
        const size_t total_constraints = input.constraints.size();
        input.constraints.swap(local_constraints);
        // Fill in our shard mapping and local shard info
        input.shard_mapping = repl_ctx->shard_manager->shard_mapping;
        input.local_shard = repl_ctx->owner_shard->shard_id;
        // Update the outputs
        output.task_processors.resize(input.tasks.size());
        output.constraint_mappings.resize(input.constraints.size());
        output.weights.resize(input.constraints.size());
        // Now we can do the mapper call
        mapper->invoke_map_must_epoch(this, input, output);
        // Now we need to exchange our mapping decisions between all the shards
        legion_assert(mapping_exchange == nullptr);
        legion_assert(mapping_collective_id > 0);
        mapping_exchange =
            new MustEpochMappingExchange(repl_ctx, mapping_collective_id);
        mapping_exchange->exchange_must_epoch_mappings(
            repl_ctx->owner_shard->shard_id,
            repl_ctx->shard_manager->total_shards, total_constraints,
            input.tasks, all_tasks, output.task_processors,
            original_constraint_indexes, output.constraint_mappings,
            output.weights, *get_acquired_instances_ref());
      }
      else
      {
        legion_assert(mapping_broadcast == nullptr);
        legion_assert(mapping_collective_id > 0);
        mapping_broadcast = new MustEpochMappingBroadcast(
            repl_ctx, 0 /*owner shard*/, mapping_collective_id);
        // Do the mapper call on shard 0 and then broadcast the results
        if (repl_ctx->owner_shard->shard_id == 0)
        {
          mapper->invoke_map_must_epoch(this, input, output);
          mapping_broadcast->broadcast(
              output.task_processors, output.constraint_mappings);
        }
        else
          mapping_broadcast->receive_results(
              output.task_processors, original_constraint_indexes,
              output.constraint_mappings, *get_acquired_instances_ref());
      }
      // No need to do any checks, the base class handles that
      return mapper;
    }

    //--------------------------------------------------------------------------
    RtEvent ReplMustEpochOp::map_and_distribute(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(single_tasks.size() == mapping_dependences.size());
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      for (SingleTask* task : shard_single_tasks)
        mapped_events.emplace(
            std::make_pair(task->index_point, Runtime::create_rt_user_event()));
      // Exchange these to get them in flight
      MustEpochDependenceExchange dependence_exchange(
          dependence_exchange_id, repl_ctx, mapped_events);
      dependence_exchange.perform_collective_async();

      std::vector<RtEvent> tasks_all_mapped;
      std::vector<ApEvent> tasks_all_complete;
      tasks_all_mapped.reserve(indiv_tasks.size() + index_tasks.size());
      tasks_all_complete.reserve(indiv_tasks.size() + index_tasks.size());
      // Once all the tasks have been initialized we can defer
      // our all mapped event on all their all mapped events
      for (IndividualTask* task : indiv_tasks)
      {
        tasks_all_mapped.emplace_back(task->get_mapped_event());
        tasks_all_complete.emplace_back(task->get_completion_event());
        if (shard_single_tasks.find(task) != shard_single_tasks.end())
          remaining_resource_returns++;
      }
      for (IndexTask* task : index_tasks)
      {
        tasks_all_mapped.emplace_back(task->get_mapped_event());
        tasks_all_complete.emplace_back(task->get_completion_event());
      }
      // Start the exchange for the mapped and completion events
      MustEpochCompletionExchange completion_exchange(
          completion_exchange_id, repl_ctx, tasks_all_mapped,
          tasks_all_complete);
      completion_exchange.perform_collective_async();
      // Need to count remaining resource returns for slices too
      for (SliceTask* slice : slice_tasks)
      {
        // Check to see if we either do or not own this slice
        // We currently do not support mixed slices for which
        // we only own some of the points
        bool contains_any = false;
        bool contains_all = true;
        for (PointTask* task : slice->points)
        {
          if (shard_single_tasks.find(task) != shard_single_tasks.end())
            contains_any = true;
          else if (contains_all)
          {
            contains_all = false;
            if (contains_any)  // At this point we have all the answers
              break;
          }
        }
        if (!contains_any)
          continue;
        if (!contains_all)
        {
          Processor mapper_proc = parent_ctx->get_executing_processor();
          MapperManager* mapper = runtime->find_mapper(mapper_proc, map_id);
          {
            Fatal fatal;
            fatal << "Mapper " << mapper->get_mapper_name()
                  << " specified a slice for a must epoch launch in control "
                     "replicated task "
                  << parent_ctx->get_task_name() << " (UID "
                  << parent_ctx->get_unique_id()
                  << ") for which not all the points mapped to the same shard. "
                     "Legion does not "
                  << "currently support this use case. Please specify slices "
                     "and a sharding function to "
                  << "ensure that all the points in a slice are owned by the "
                     "same shard.";
            fatal.raise();
          }
          remaining_resource_returns++;
        }
        remaining_resource_returns++;
      }
      // Trigger this if we're not expecting to see any returns
      if (remaining_resource_returns == 0)
        runtime->phase_barrier_arrive(resource_return_barrier, 1 /*count*/);
      // Update the remaining concurrent points
      if (shard_single_tasks.empty())
      {
        // Still need to participate in things even if we have no local points
        finalize_concurrent_mapped();
        finish_concurrent_allreduce();
        collective_exchange =
            new AllReduceCollective<MaxReduction<uint64_t>, false>(
                repl_ctx, collective_exchange_id);
        collective_exchange->async_all_reduce(collective_lamport_clock);
        AutoLock o_lock(op_lock);
        commit_preconditions.insert(collective_exchange->get_done_event());
      }
      else
      {
        remaining_mapped_events.store(shard_single_tasks.size());
        remaining_collective_unbound_points = shard_single_tasks.size();
        remaining_concurrent_mapped = shard_single_tasks.size();
        remaining_concurrent_points = shard_single_tasks.size();
      }
      // Wait for the point-wise exchange to be done and then launch
      // of all our local single tasks
      dependence_exchange.perform_collective_wait();
      // For correctness we still have to abide by the mapping dependences
      // computed on the individual tasks while we are mapping them
      for (unsigned idx = 0; idx < single_tasks.size(); idx++)
      {
        // Check to see if it is one of the ones that we own
        if (shard_single_tasks.find(single_tasks[idx]) ==
            shard_single_tasks.end())
        {
          // We don't own this point
          // We still need to do some work for individual tasks
          // to exchange versioning information, but no such
          // work is necessary for point tasks
          SingleTask* task = single_tasks[idx];
          task->shard_off(mapped_events[task->index_point]);
          continue;
        }
        // Figure out our preconditions
        std::set<RtEvent> preconditions;
        for (const unsigned& it : mapping_dependences[idx])
        {
          legion_assert(it < idx);
          preconditions.insert(mapped_events[single_tasks[it]->index_point]);
        }
        RtEvent precondition;
        if (!preconditions.empty())
          precondition = Runtime::merge_events(preconditions);
        if (precondition.exists() && !precondition.has_triggered())
          single_tasks[idx]->defer_perform_mapping(precondition, this);
        else if (
            single_tasks[idx]->perform_mapping(this) &&
            single_tasks[idx]->distribute_task())
          single_tasks[idx]->launch_task();
      }
      return completion_exchange.finish_exchange(this);
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::trigger_prepipeline_stage(void)
    //--------------------------------------------------------------------------
    {
      Processor mapper_proc = parent_ctx->get_executing_processor();
      MapperManager* mapper = runtime->find_mapper(mapper_proc, map_id);
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      // Select our sharding functor and then do the base call
      this->individual_tasks.resize(indiv_tasks.size());
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
        this->individual_tasks[idx] = indiv_tasks[idx];
      this->index_space_tasks.resize(index_tasks.size());
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
        this->index_space_tasks[idx] = index_tasks[idx];
      Mapper::SelectShardingFunctorInput sharding_input;
      sharding_input.shard_mapping = repl_ctx->shard_manager->shard_mapping;
      Mapper::MustEpochShardingFunctorOutput sharding_output;
      sharding_output.chosen_functor = std::numeric_limits<ShardingID>::max();
      sharding_output.collective_map_must_epoch_call = false;
      mapper->invoke_must_epoch_select_sharding_functor(
          this, sharding_input, sharding_output);
      // We can clear these now that we don't need them anymore
      individual_tasks.clear();
      index_space_tasks.clear();
      // Check that we have a sharding ID
      if (sharding_output.chosen_functor ==
          std::numeric_limits<ShardingID>::max())
      {
        Error error(LEGION_MAPPER_EXCEPTION);
        error << "Invalid mapper output from invocation of 'map_must_epoch' on "
                 "mapper "
              << mapper->get_mapper_name()
              << ". Mapper failed to specify a valid sharding ID "
              << "for a must epoch operation in control replicated context of "
                 "task "
              << repl_ctx->get_task_name() << " (UID "
              << repl_ctx->get_unique_id() << ").";
        error.raise();
      }
      this->sharding_functor = sharding_output.chosen_functor;
      this->collective_map_must_epoch_call =
          sharding_output.collective_map_must_epoch_call;
      legion_assert(sharding_function == nullptr);
      // Check that the sharding IDs are all the same
      if (runtime->safe_mapper)
      {
        legion_assert(sharding_collective != nullptr);
        // Contribute the result
        sharding_collective->contribute(this->sharding_functor);
        if (sharding_collective->is_target() &&
            !sharding_collective->validate(this->sharding_functor))
        {
          Error error(LEGION_MAPPER_EXCEPTION);
          error << "Mapper " << *mapper << " chose different sharding "
                << "functions for must epoch launch in " << *parent_ctx;
          error.raise();
        }
      }
      ReplFutureMapImpl* impl =
          legion_safe_cast<ReplFutureMapImpl*>(result_map.impl);
      // Set the future map sharding functor
      sharding_function =
          repl_ctx->shard_manager->find_sharding_function(sharding_functor);
      impl->set_sharding_function(sharding_function);
      // Set the sharding functor for all the point and index tasks too
      for (unsigned idx = 0; idx < indiv_tasks.size(); idx++)
      {
        ReplIndividualTask* task =
            static_cast<ReplIndividualTask*>(indiv_tasks[idx]);
        task->set_sharding_function(sharding_functor, sharding_function);
      }
      for (unsigned idx = 0; idx < index_tasks.size(); idx++)
      {
        ReplIndexTask* task = static_cast<ReplIndexTask*>(index_tasks[idx]);
        task->set_sharding_function(sharding_functor, sharding_function);
      }
      MustEpochOp::trigger_prepipeline_stage();
    }

    //--------------------------------------------------------------------------
    uint64_t ReplMustEpochOp::collective_lamport_allreduce(
        uint64_t lamport_clock, bool need_result)
    //--------------------------------------------------------------------------
    {
      ReplicateContext* repl_ctx =
          legion_safe_cast<ReplicateContext*>(parent_ctx);
      {
        AutoLock o_lock(op_lock);
        if (collective_lamport_clock < lamport_clock)
          collective_lamport_clock = lamport_clock;
        legion_assert(remaining_collective_unbound_points > 0);
        if (--remaining_collective_unbound_points > 0)
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
    void ReplMustEpochOp::finalize_concurrent_mapped(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(concurrent_mapped_barrier.exists());
      runtime->phase_barrier_arrive(
          concurrent_mapped_barrier, 1 /*count*/,
          Runtime::merge_events(concurrent_preconditions));
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::finish_concurrent_allreduce(void)
    //--------------------------------------------------------------------------
    {
      concurrent_exchange->perform_concurrent_allreduce(
          concurrent_tasks, concurrent_slices, concurrent_lamport_clock,
          concurrent_poisoned);
    }

    //--------------------------------------------------------------------------
    bool ReplMustEpochOp::has_return_resources(void) const
    //--------------------------------------------------------------------------
    {
      return !(
          created_regions.empty() && local_regions.empty() &&
          created_fields.empty() && local_fields.empty() &&
          created_field_spaces.empty() && created_index_spaces.empty() &&
          created_index_partitions.empty() && deleted_regions.empty() &&
          deleted_fields.empty() && deleted_field_spaces.empty() &&
          latent_field_spaces.empty() && deleted_index_spaces.empty() &&
          deleted_index_partitions.empty());
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::receive_resources(
        uint64_t return_index, std::map<LogicalRegion, unsigned>& created_regs,
        std::vector<DeletedRegion>& deleted_regs,
        std::set<std::pair<FieldSpace, FieldID> >& created_fids,
        std::vector<DeletedField>& deleted_fids,
        std::map<FieldSpace, unsigned>& created_fs,
        std::map<FieldSpace, std::set<LogicalRegion> >& latent_fs,
        std::vector<DeletedFieldSpace>& deleted_fs,
        std::map<IndexSpace, unsigned>& created_is,
        std::vector<DeletedIndexSpace>& deleted_is,
        std::map<IndexPartition, unsigned>& created_partitions,
        std::vector<DeletedPartition>& deleted_partitions,
        std::set<RtEvent>& preconditions)
    //--------------------------------------------------------------------------
    {
      // Wait until we've received all the resources before handing them
      // back to the enclosing parent context
      {
        AutoLock o_lock(op_lock);
        merge_received_resources(
            created_regs, deleted_regs, created_fids, deleted_fids, created_fs,
            latent_fs, deleted_fs, created_is, deleted_is, created_partitions,
            deleted_partitions);
        legion_assert(remaining_resource_returns > 0);
        if (--remaining_resource_returns > 0)
          return;
      }
      // Make sure the other shards have received all their returns too
      runtime->phase_barrier_arrive(resource_return_barrier, 1 /*count*/);
      if (!has_return_resources())
        return;
      if (!resource_return_barrier.has_triggered())
      {
        DeferMustEpochReturnResourcesArgs args(this);
        runtime->issue_runtime_meta_task(
            args, LG_THROUGHPUT_DEFERRED_PRIORITY, resource_return_barrier);
        preconditions.insert(args.done);
        return;
      }
      // If we get here then we can finally do the return to the parent context
      // because we've received resources from all of our constituent operations
      return_resources(parent_ctx, context_index, preconditions);
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::DeferMustEpochReturnResourcesArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      std::set<RtEvent> preconditions;
      op->return_resources(
          op->get_context(), op->get_context_index(), preconditions);
      if (!preconditions.empty())
        Runtime::trigger_event(done, Runtime::merge_events(preconditions));
      else
        Runtime::trigger_event(done);
    }

    //--------------------------------------------------------------------------
    void ReplMustEpochOp::initialize_replication(ReplicateContext* ctx)
    //--------------------------------------------------------------------------
    {
      legion_assert(mapping_collective_id == 0);
      legion_assert(collective_exchange_id == 0);
      legion_assert(mapping_broadcast == nullptr);
      legion_assert(mapping_exchange == nullptr);
      // We can't actually make a collective for the mapping yet because we
      // don't know if we are going to broadcast or exchange so we just get
      // a collective ID that we will use later
      mapping_collective_id = ctx->get_next_collective_index(COLLECTIVE_LOC_58);
      dependence_exchange_id =
          ctx->get_next_collective_index(COLLECTIVE_LOC_70);
      completion_exchange_id =
          ctx->get_next_collective_index(COLLECTIVE_LOC_73);
      collective_exchange_id =
          ctx->get_next_collective_index(COLLECTIVE_LOC_107);
      concurrent_exchange = new ConcurrentAllreduce(COLLECTIVE_LOC_69, ctx);
      resource_return_barrier = ctx->get_next_resource_return_barrier();
    }

    //--------------------------------------------------------------------------
    Domain ReplMustEpochOp::get_shard_domain(void) const
    //--------------------------------------------------------------------------
    {
      if (sharding_space.exists())
      {
        Domain shard_domain;
        runtime->find_domain(sharding_space, shard_domain);
        return shard_domain;
      }
      else
        return launch_domain;
    }

  }  // namespace Internal
}  // namespace Legion
