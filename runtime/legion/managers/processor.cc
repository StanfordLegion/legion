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

#include "legion/managers/processor.h"
#include "legion/contexts/inner.h"
#include "legion/kernel/runtime.h"
#include "legion/managers/mapper.h"

namespace Legion {
  namespace Internal {

    /////////////////////////////////////////////////////////////
    // Processor Manager
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ProcessorManager::ProcessorManager(
        Processor proc, Processor::Kind kind, unsigned def_mappers,
        bool no_steal, bool replay)
      : local_proc(proc), proc_kind(kind), stealing_disabled(no_steal),
        replay_execution(replay), next_local_index(0),
        task_scheduler_enabled(false), outstanding_task_scheduler(false),
        total_active_contexts(0), total_active_mappers(0),
        total_progress_tasks(0), concurrent_lamport_clock(0),
        ready_concurrent_tasks(0), outstanding_concurrent_task(false)
    //--------------------------------------------------------------------------
    {
      context_states.resize(LEGION_DEFAULT_CONTEXTS);
      // Find our set of visible memories
      Machine::MemoryQuery vis_mems(runtime->machine);
      vis_mems.has_affinity_to(proc);
      vis_mems.has_capacity(1 /*at least one byte*/);
      for (Machine::MemoryQuery::iterator it = vis_mems.begin();
           it != vis_mems.end(); it++)
      {
        Realm::Machine::AffinityDetails affinity;
        runtime->machine.has_affinity(proc, *it, &affinity);
        visible_memories[*it] = affinity.bandwidth;
      }
    }

    //--------------------------------------------------------------------------
    ProcessorManager::~ProcessorManager(void)
    //--------------------------------------------------------------------------
    {
      mapper_states.clear();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::prepare_for_shutdown(void)
    //--------------------------------------------------------------------------
    {
      for (std::pair<const MapperID, MapperManager*>& entry : mappers)
      {
        entry.second->prepare_for_shutdown();
        if (entry.second->remove_reference())
          delete entry.second;
      }
      mappers.clear();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_mapper(
        MapperID mid, MapperManager* m, bool check, bool skip_replay)
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are doing replay execution
      if (!skip_replay && replay_execution)
        return;
      if (check && (mid == 0))
      {
        Error err(LEGION_INTERFACE_EXCEPTION);
        err << "Invalid mapping ID. ID 0 is reserved.";
        err.raise();
      }
      if (check && !inside_registration_callback)
      {
        Warning warn;
        warn
            << "Mapper " << m->get_mapper_name() << " (ID " << mid
            << ") was dynamically registered outside of a "
            << "registration callback invocation. In the near future this will "
            << "become an error in order to support task subprocesses. Please "
            << "use 'perform_registration_callback' to generate a callback "
            << "where it will be safe to perform dynamic registrations.";
        warn.raise();
      }
      m->add_reference();
      AutoLock m_lock(mapper_lock);
      std::map<MapperID, MapperManager*>::iterator finder = mappers.find(mid);
      if (finder != mappers.end())
      {
        finder->second->prepare_for_shutdown();
        if (finder->second->remove_reference())
          delete finder->second;
        finder->second = m;
      }
      else
      {
        mappers.emplace(mid, m);
        AutoLock q_lock(queue_lock);
        mapper_states[mid] = MapperState();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::replace_default_mapper(MapperManager* m)
    //--------------------------------------------------------------------------
    {
      // Don't do this if we are doing replay execution
      if (replay_execution)
        return;
      if (!inside_registration_callback)
      {
        Warning warn;
        warn << "Replacing default mapper with " << m->get_mapper_name()
             << " was dynamically performed "
             << "outside of a registration callback invocation. In the near "
             << "future this will become an error in order to support task "
             << "subprocesses. Please use 'perform_registration_callback' to "
             << "generate a callback where it will be safe to perform dynamic "
             << "registrations.";
        warn.raise();
      }
      m->add_reference();
      AutoLock m_lock(mapper_lock);
      std::map<MapperID, MapperManager*>::iterator finder = mappers.find(0);
      if (finder != mappers.end())
      {
        finder->second->prepare_for_shutdown();
        if (finder->second->remove_reference())
          delete finder->second;
        finder->second = m;
      }
      else
      {
        mappers.emplace(0, m);
        AutoLock q_lock(queue_lock);
        mapper_states[0] = MapperState();
      }
    }

    //--------------------------------------------------------------------------
    MapperManager* ProcessorManager::find_mapper(MapperID mid) const
    //--------------------------------------------------------------------------
    {
      // Easy case if we are doing replay execution
      if (replay_execution)
      {
        std::map<MapperID, MapperManager*>::const_iterator finder =
            mappers.find(0);
        legion_assert(finder != mappers.end());
        return finder->second;
      }
      AutoLock m_lock(mapper_lock, false /*exclusive*/);
      MapperManager* result = nullptr;
      // We've got the lock, so do the operation
      std::map<MapperID, MapperManager*>::const_iterator finder =
          mappers.find(mid);
      if (finder != mappers.end())
        result = finder->second;
      return result;
    }

    //--------------------------------------------------------------------------
    bool ProcessorManager::has_non_default_mapper(void) const
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock, false /*exclusive*/);
      for (const std::pair<const MapperID, MapperManager*>& entry : mappers)
        if (!entry.second->is_default_mapper)
          return true;
      return false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::perform_scheduling(void)
    //--------------------------------------------------------------------------
    {
      perform_mapping_operations();
      // Now re-take the lock and re-check the condition to see
      // if the next scheduling task should be launched
      AutoLock q_lock(queue_lock);
      legion_assert(outstanding_task_scheduler);
      // If the task scheduler is enabled launch ourselves again
      if (task_scheduler_enabled)
      {
        SchedulerArgs sched_args(this);
        // If we need to recursively run the scheduler then we do so with
        // a lower priority than other meta-tasks to ensure that those other
        // meta tasks can continue to make forward progress and the scheduler
        // cannot starve other tasks.
        runtime->issue_runtime_meta_task(
            sched_args, LG_THROUGHPUT_WORK_PRIORITY);
      }
      else
        outstanding_task_scheduler = false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::launch_task_scheduler(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!outstanding_task_scheduler);
      outstanding_task_scheduler = true;
      SchedulerArgs sched_args(this);
      // This is waking the scheduler up so give it higher priority in
      // order to ensure that we can get tasks mapped and running sooner
      runtime->issue_runtime_meta_task(sched_args, LG_LATENCY_WORK_PRIORITY);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::SchedulerArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      manager->perform_scheduling();
#ifdef LEGION_TRACE_ALLOCATION
      unsigned long long trace_count =
          runtime->allocation_tracing_count.fetch_add(1);
      if ((trace_count % LEGION_TRACE_ALLOCATION_FREQUENCY) == 0)
        runtime->dump_allocation_info();
#endif
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::notify_deferred_mapper(
        MapperID map_id, RtEvent deferred_event)
    //--------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      MapperState& state = mapper_states[map_id];
      // Check to see if the deferral event matches the one that we have
      if (state.deferral_event == deferred_event)
      {
        // Now we can clear it
        state.deferral_event = RtEvent::NO_RT_EVENT;
        // And if we still have tasks, reactivate the mapper
        if (!state.ready_queue.empty())
          increment_active_mappers();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::DeferMapperSchedulerArgs::execute(void) const
    //--------------------------------------------------------------------------
    {
      proxy_this->notify_deferred_mapper(map_id, deferral_event);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::activate_context(InnerContext* context)
    //--------------------------------------------------------------------------
    {
      ContextID ctx_id = context->get_logical_tree_context();
      AutoLock q_lock(queue_lock);
      ContextState& state = context_states[ctx_id];
      legion_assert(!state.active);
      state.active = true;
      if (state.owned_tasks > 0)
        increment_active_contexts();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::deactivate_context(InnerContext* context)
    //--------------------------------------------------------------------------
    {
      ContextID ctx_id = context->get_logical_tree_context();
      // We can do this without holding the lock because we know
      // the size of this vector is fixed
      AutoLock q_lock(queue_lock);
      ContextState& state = context_states[ctx_id];
      legion_assert(state.active);
      state.active = false;
      if (state.owned_tasks > 0)
        decrement_active_contexts();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::update_max_context_count(unsigned max_contexts)
    //--------------------------------------------------------------------------
    {
      AutoLock q_lock(queue_lock);
      context_states.resize(max_contexts);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::increment_active_contexts(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      if (!task_scheduler_enabled && (total_active_contexts == 0) &&
          (total_progress_tasks == 0) && (total_active_mappers > 0))
      {
        task_scheduler_enabled = true;
        if (!outstanding_task_scheduler)
          launch_task_scheduler();
      }
      total_active_contexts++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::decrement_active_contexts(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      legion_assert(total_active_contexts > 0);
      total_active_contexts--;
      if ((total_active_contexts == 0) && (total_progress_tasks == 0))
        task_scheduler_enabled = false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::increment_active_mappers(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      if (!task_scheduler_enabled && (total_active_mappers == 0) &&
          ((total_active_contexts > 0) || (total_progress_tasks > 0)))
      {
        task_scheduler_enabled = true;
        if (!outstanding_task_scheduler)
          launch_task_scheduler();
      }
      total_active_mappers++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::decrement_active_mappers(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      legion_assert(total_active_mappers > 0);
      total_active_mappers--;
      if (total_active_mappers == 0)
        task_scheduler_enabled = false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::increment_progress_tasks(void)
    //--------------------------------------------------------------------------
    {
      // Better be called while holding the queue lock
      if (!task_scheduler_enabled && (total_active_contexts == 0) &&
          (total_progress_tasks == 0) && (total_active_mappers > 0))
      {
        task_scheduler_enabled = true;
        if (!outstanding_task_scheduler)
          launch_task_scheduler();
      }
      total_progress_tasks++;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::decrement_progress_tasks(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(total_progress_tasks > 0);
      total_progress_tasks--;
      if ((total_active_contexts == 0) && (total_progress_tasks == 0))
        task_scheduler_enabled = false;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::process_steal_request(
        Processor thief, const std::vector<MapperID>& thieves)
    //--------------------------------------------------------------------------
    {
      // Iterate over the task descriptions, asking the appropriate mapper
      // whether we can steal the task
      std::vector<SingleTask*> stolen;
      std::vector<MapperID> successful_thieves;
      for (const MapperID& stealer : thieves)
      {
        // Handle a race condition here where some processors can
        // issue steal requests to another processor before the mappers
        // have been initialized on that processor.  There's no
        // correctness problem for ignoring a steal request so just do that.
        MapperManager* mapper = find_mapper(stealer);
        if (mapper == nullptr)
          continue;
        Mapper::StealRequestInput input;
        {
          // Wait until we can exclusive access to the ready queue
          RtEvent queue_copy_ready;
          // Pull out the current tasks for this mapping operation
          // Need to iterate until we get access to the queue
          do {
            if (queue_copy_ready.exists() && !queue_copy_ready.has_triggered())
            {
              queue_copy_ready.wait();
              queue_copy_ready = RtEvent::NO_RT_EVENT;
            }
            AutoLock q_lock(queue_lock);
            MapperState& map_state = mapper_states[stealer];
            if (!map_state.queue_guard)
            {
              // If we don't have a deferral event then grab our
              // ready queue of tasks so we can try to map them
              // this will also prevent them from being stolen
              if (!map_state.ready_queue.empty())
              {
                for (SingleTask* task : map_state.ready_queue)
                  if (task->is_stealable() && !task->is_origin_mapped())
                    input.stealable_tasks.emplace_back(task);
                // Set the queue guard so no one else tries to
                // read the ready queue while we've checked it out
                if (!input.stealable_tasks.empty())
                  map_state.queue_guard = true;
              }
            }
            else
            {
              // Make an event if necessary
              if (!map_state.queue_waiter.exists())
                map_state.queue_waiter = Runtime::create_rt_user_event();
              // Record that we need to wait on it
              queue_copy_ready = map_state.queue_waiter;
            }
          } while (queue_copy_ready.exists());
        }
        if (input.stealable_tasks.empty())
          continue;
        input.thief_proc = thief;
        Mapper::StealRequestOutput output;
        // Ask the mapper what it wants to allow be stolen
        if (!input.stealable_tasks.empty())
          mapper->invoke_permit_steal_request(input, output);
        // See which tasks we can succesfully steal
        std::vector<SingleTask*> local_stolen;
        {
          // Retake the lock, put any tasks still in the ready queue
          // back into the queue and remove the queue guard
          AutoLock q_lock(queue_lock);
          MapperState& map_state = mapper_states[stealer];
          legion_assert(map_state.queue_guard);
          std::list<SingleTask*>& rqueue = map_state.ready_queue;
          for (std::list<SingleTask*>::iterator it = rqueue.begin();
               it != rqueue.end();
               /*nothing*/)
          {
            if (output.stolen_tasks.find(*it) != output.stolen_tasks.end())
            {
              const ContextID ctx_id =
                  (*it)->get_context()->get_logical_tree_context();
              ContextState& state = context_states[ctx_id];
              legion_assert(state.owned_tasks > 0);
              state.owned_tasks--;
              if (state.active && (state.owned_tasks == 0))
                decrement_active_contexts();
              if ((*it)->is_forward_progress_task())
                decrement_progress_tasks();
              (*it)->mark_stolen();
              local_stolen.emplace_back(*it);
              it = rqueue.erase(it);
            }
            else
              it++;
          }
          if (rqueue.empty())
          {
            if (map_state.deferral_event.exists())
              map_state.deferral_event = RtEvent::NO_RT_EVENT;
            else
              decrement_active_mappers();
          }
          // Remove the queue guard
          map_state.queue_guard = false;
          if (map_state.queue_waiter.exists())
          {
            Runtime::trigger_event(map_state.queue_waiter);
            map_state.queue_waiter = RtUserEvent::NO_RT_USER_EVENT;
          }
        }
        if (!local_stolen.empty())
        {
          successful_thieves.emplace_back(stealer);
          for (SingleTask* task : local_stolen)
            task->deactivate_outstanding_task();
          if (stolen.empty())
            stolen.swap(local_stolen);
          else
            stolen.insert(
                stolen.end(), local_stolen.begin(), local_stolen.end());
        }
        else
          mapper->process_failed_steal(thief);
      }
      if (!stolen.empty())
      {
        runtime->send_tasks(thief, stolen);
        // Also have to send advertisements to the mappers that
        // successfully stole so they know that they can try again
        std::set<Processor> thief_set;
        thief_set.insert(thief);
        for (const MapperID& thief : successful_thieves)
          runtime->send_advertisements(thief_set, thief, local_proc);
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::process_advertisement(
        Processor advertiser, MapperID mid)
    //--------------------------------------------------------------------------
    {
      MapperManager* mapper = find_mapper(mid);
      mapper->process_advertisement(advertiser);
      // See if this mapper would like to try stealing again
      std::multimap<Processor, MapperID> stealing_targets;
      mapper->perform_stealing(stealing_targets);
      if (!stealing_targets.empty())
        runtime->send_steal_request(stealing_targets, local_proc);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::add_to_ready_queue(SingleTask* task)
    //--------------------------------------------------------------------------
    {
      legion_assert(task != nullptr);
      // have to do this when we are not holding the lock
      task->activate_outstanding_task();
      // Check to see if this task is a task that must map in order to
      // guarantee forward progress
      const bool forward_progress_task = task->is_forward_progress_task();
      // We can do this without holding the lock because the
      // vector is of a fixed size
      ContextID ctx_id = task->get_context()->get_logical_tree_context();
      AutoLock q_lock(queue_lock);
      legion_assert(mapper_states.find(task->map_id) != mapper_states.end());
      // Update the state for the context
      ContextState& state = context_states[ctx_id];
      if (state.active && (state.owned_tasks == 0))
        increment_active_contexts();
      state.owned_tasks++;
      // Also update the queue for the mapper
      MapperState& map_state = mapper_states[task->map_id];
      if (map_state.ready_queue.empty() || map_state.deferral_event.exists())
      {
        // Clear our deferral event since we are changing state
        map_state.deferral_event = RtEvent::NO_RT_EVENT;
        increment_active_mappers();
      }
      map_state.ready_queue.emplace_back(task);
      if (map_state.queue_guard)
        map_state.queue_dirty = true;
      // Finally if this is a progress task increment it
      if (forward_progress_task)
        increment_progress_tasks();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::find_visible_memories(
        std::set<Memory>& visible) const
    //--------------------------------------------------------------------------
    {
      for (const std::pair<const Memory, size_t>& memory : visible_memories)
        visible.insert(memory.first);
    }

    //--------------------------------------------------------------------------
    Memory ProcessorManager::find_best_visible_memory(Memory::Kind kind) const
    //--------------------------------------------------------------------------
    {
      size_t affinity = 0;
      Memory result = Memory::NO_MEMORY;
      for (const std::pair<const Memory, size_t>& it : visible_memories)
      {
        if (it.first.kind() != kind)
          continue;
        if (it.second < affinity)
          continue;
        result = it.first;
        affinity = it.second;
      }
      return result;
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::order_concurrent_task_launch(
        SingleTask* task, ApEvent precondition, ApUserEvent ready,
        VariantID vid)
    //--------------------------------------------------------------------------
    {
      uint64_t lamport_clock = 0;
      {
        AutoLock c_lock(concurrent_lock);
        legion_assert(concurrent_tasks.find(task) == concurrent_tasks.end());
        lamport_clock = concurrent_lamport_clock++;
        concurrent_tasks.insert(std::make_pair(
            task, ConcurrentState(lamport_clock, precondition, ready)));
      }
      // Check to see if the precondition event was poisoned
      bool poisoned = false;
      legion_no_skip_assert(precondition.has_triggered_faultaware(poisoned));
      // Tell the task to compute the max all-reduce of lamport clocks
      task->concurrent_allreduce(this, lamport_clock, vid, poisoned);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::finalize_concurrent_task_order(
        SingleTask* task, uint64_t lamport_clock, bool poisoned)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(concurrent_lock);
      std::map<SingleTask*, ConcurrentState>::iterator finder =
          concurrent_tasks.find(task);
      legion_assert(finder != concurrent_tasks.end());
      legion_assert(!finder->second.max);
      legion_assert(finder->second.lamport_clock <= lamport_clock);
      if (concurrent_lamport_clock <= lamport_clock)
        concurrent_lamport_clock = lamport_clock + 1;
      if (poisoned)
      {
        Runtime::poison_event(finder->second.ready);
        concurrent_tasks.erase(finder);
      }
      else
      {
        finder->second.lamport_clock = lamport_clock;
        finder->second.max = true;
        ready_concurrent_tasks++;
        if (!outstanding_concurrent_task)
          start_next_concurrent_task();
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::end_concurrent_task(void)
    //--------------------------------------------------------------------------
    {
      AutoLock c_lock(concurrent_lock);
      legion_assert(outstanding_concurrent_task);
      outstanding_concurrent_task = false;
      if (ready_concurrent_tasks > 0)
        start_next_concurrent_task();
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::start_next_concurrent_task(void)
    //--------------------------------------------------------------------------
    {
      legion_assert(!concurrent_tasks.empty());
      legion_assert(!outstanding_concurrent_task);
      legion_assert(ready_concurrent_tasks > 0);
      // See if we can prove that there is a task that is safe to start
      uint64_t min_next = std::numeric_limits<uint64_t>::max();
      uint64_t min_pending = std::numeric_limits<uint64_t>::max();
      SingleTask* next = nullptr;
      TaskTreeCoordinates next_coords;
      for (const std::pair<SingleTask* const, ConcurrentState>& it :
           concurrent_tasks)
      {
        if (it.second.max)
        {
          if (next != nullptr)
          {
            // Compare the lamport clocks
            if (it.second.lamport_clock < min_next)
            {
              next = it.first;
              next_coords.clear();
              min_next = it.second.lamport_clock;
            }
            else if (min_next == it.second.lamport_clock)
            {
              // Very bad case, same min of max all-reduce of clocks
              // Resolve this conflict based on task tree coordinates
              TaskTreeCoordinates it_coords;
              if (next_coords.empty())
                next->compute_task_tree_coordinates(next_coords);
              it.first->compute_task_tree_coordinates(it_coords);
              const size_t lower_bound =
                  std::min(next_coords.size(), it_coords.size());
              bool equal = true;
              for (unsigned idx = 0; idx < lower_bound; idx++)
              {
                const ContextCoordinate& c1 = next_coords[idx];
                const ContextCoordinate& c2 = it_coords[idx];
                if (c1.context_index == c2.context_index)
                {
                  if (c2.index_point < c1.index_point)
                  {
                    next = it.first;
                    next_coords.swap(it_coords);
                  }
                  else if (c1.index_point == c2.index_point)
                    continue;
                }
                else if (c2.context_index < c1.context_index)
                {
                  next = it.first;
                  next_coords.swap(it_coords);
                }
                equal = false;
                break;
              }
              if (equal)
              {
                legion_assert(next_coords.size() != it_coords.size());
                if (it_coords.size() < next_coords.size())
                {
                  next = it.first;
                  next_coords.swap(it_coords);
                }
              }
            }
          }
          else
          {
            next = it.first;
            min_next = it.second.lamport_clock;
          }
        }
        else if (it.second.lamport_clock < min_pending)
          min_pending = it.second.lamport_clock;
      }
      // If all the pending tasks with lamport clocks are
      // larger than our max lamport clock of the next task
      // to launch then we know they won't ever come before it
      // so we can issue our next task now, otherwise we'll need
      // to wait until those pending lamport clocks are done
      if (min_next < min_pending)
      {
        std::map<SingleTask*, ConcurrentState>::iterator finder =
            concurrent_tasks.find(next);
        legion_assert(finder != concurrent_tasks.end());
        // Trigger the ready event with the precondition to keep
        // tools like Legion Spy happy even though we know that
        // the precondition event has already triggered
        Runtime::trigger_event_untraced(
            finder->second.ready, finder->second.precondition);
        concurrent_tasks.erase(finder);
        ready_concurrent_tasks--;
        outstanding_concurrent_task = true;
      }
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::perform_mapping_operations(void)
    //--------------------------------------------------------------------------
    {
      std::multimap<Processor, MapperID> stealing_targets;
      std::vector<MapperID> mappers_with_stealable_work;
      std::vector<std::pair<MapperID, MapperManager*> > current_mappers;
      // Take a snapshot of our current mappers
      {
        AutoLock m_lock(mapper_lock, false /*exclusive*/);
        // Fast path for no deferred mappers
        current_mappers.reserve(mappers.size());
        for (const std::pair<const MapperID, MapperManager*>& it : mappers)
          current_mappers.emplace_back(it);
      }
      for (const std::pair<MapperID, MapperManager*>& it : current_mappers)
      {
        const MapperID map_id = it.first;
        MapperManager* const mapper = it.second;
        Mapper::SelectMappingInput input;
        input.processor = local_proc;
        {
          RtEvent input_ready;
          // Pull out the current tasks for this mapping operation
          // Need to iterate until we get access to the queue
          do {
            if (input_ready.exists() && !input_ready.has_triggered())
            {
              input_ready.wait();
              input_ready = RtEvent::NO_RT_EVENT;
            }
            AutoLock q_lock(queue_lock);
            MapperState& map_state = mapper_states[map_id];
            if (!map_state.queue_guard)
            {
              // If we don't have a deferral event then grab our
              // ready queue of tasks so we can try to map them
              // this will also prevent them from being stolen
              if (!map_state.deferral_event.exists() &&
                  !map_state.ready_queue.empty())
              {
                // Only ask the mapper about ready tasks that have
                // active contexts that we should keep mapping
                for (SingleTask* task : map_state.ready_queue)
                {
                  const ContextID ctx =
                      task->get_context()->get_logical_tree_context();
                  const ContextState& ctx_state = context_states[ctx];
                  if (ctx_state.active || task->is_forward_progress_task())
                    input.ready_tasks.emplace_back(task);
                }
                // Set the queue guard so no one else tries to
                // read the ready queue while we've checked it out
                if (!input.ready_tasks.empty())
                {
                  map_state.queue_guard = true;
                  map_state.queue_dirty = false;
                }
              }
            }
            else
            {
              // Make an event if necessary
              if (!map_state.queue_waiter.exists())
                map_state.queue_waiter = Runtime::create_rt_user_event();
              // Record that we need to wait on it
              input_ready = map_state.queue_waiter;
            }
          } while (input_ready.exists());
        }
        // Do this before anything else in case we don't have any tasks
        if (!stealing_disabled)
          mapper->perform_stealing(stealing_targets);
        // Nothing to do if there are no tasks on the queue
        if (input.ready_tasks.empty())
          continue;
        // Ask the mapper which tasks it would like to schedule
        Mapper::SelectMappingOutput output;
        mapper->invoke_select_tasks_to_map(input, output);
        // If we had no entry then we better have gotten a mapper event
        if (output.map_tasks.empty() && output.relocate_tasks.empty())
        {
          const RtEvent wait_on = output.deferral_event.impl;
          if (wait_on.exists())
          {
            // Put this on the list of the deferred mappers
            AutoLock q_lock(queue_lock);
            MapperState& map_state = mapper_states[map_id];
            legion_assert(!map_state.deferral_event.exists());
            legion_assert(map_state.queue_guard);
            // We have to check to see if any new tasks were added to
            // the ready queue while we were doing our mapper call, if
            // they were then we need to invoke select_tasks_to_map again
            if (!map_state.queue_dirty)
            {
              map_state.deferral_event = wait_on;
              // Decrement the number of active mappers
              decrement_active_mappers();
              // Clear the queue guard
              map_state.queue_guard = false;
              if (map_state.queue_waiter.exists())
              {
                Runtime::trigger_event(map_state.queue_waiter);
                map_state.queue_waiter = RtUserEvent::NO_RT_USER_EVENT;
              }
              // Launch a task to remove the deferred mapper
              // event when it triggers
              DeferMapperSchedulerArgs args(this, map_id, wait_on);
              // If we need to recursively run the scheduler then we do so with
              // a lower priority than other meta-tasks to ensure that those
              // other meta tasks can continue to make forward progress and the
              // scheduler cannot starve other tasks
              runtime->issue_runtime_meta_task(
                  args, LG_THROUGHPUT_WORK_PRIORITY, wait_on);
              // We can continue because there is nothing
              // left to do for this mapper
              continue;
            }
            // Otherwise we fall through to put our tasks back on the queue
            // which will lead to select_tasks_to_map being called again
          }
          else  // Very bad, error message
          {
            Error error(LEGION_MAPPER_EXCEPTION);
            error << "Mapper " << mapper->get_mapper_name()
                  << " failed to specify an output MapperEvent "
                  << "when returning from a call to 'select_tasks_to_map' "
                  << "that performed no other actions. Specifying a "
                  << "MapperEvent in such situation is necessary to avoid "
                  << "livelock conditions. Please return a "
                  << "'deferral_event' in the 'output' struct.";
            error.raise();
          }
        }
        else if (!output.relocate_tasks.empty())
        {
          for (const std::pair<const Task* const, Processor>& task :
               output.relocate_tasks)
            if (task.second.kind() == Processor::UTIL_PROC)
            {
              Error error(LEGION_MAPPER_EXCEPTION);
              error
                  << "Invalid mapper output. Mapper "
                  << mapper->get_mapper_name() << " requested that task "
                  << task.first->get_task_name() << " (UID "
                  << task.first->get_unique_id()
                  << ") be relocated to a utility processor in "
                  << "'select_tasks_to_map.' Only application processor kinds "
                  << "are permitted to be the target processor for tasks.";
              error.raise();
            }
        }
        // Figure out which tasks are to be triggered
        std::vector<SingleTask*> to_trigger;
        {
          // Retake the lock, put any tasks that the mapper didn't select
          // back on the queue and update the context states for any
          // that were selected
          AutoLock q_lock(queue_lock);
          MapperState& map_state = mapper_states[map_id];
          legion_assert(map_state.queue_guard);
          std::list<SingleTask*>& rqueue = map_state.ready_queue;
          // Iterate over the list and find any items to remove
          for (std::list<SingleTask*>::iterator it = rqueue.begin();
               it != rqueue.end();
               /*nothing*/)
          {
            if ((output.map_tasks.find(*it) != output.map_tasks.end()) ||
                (output.relocate_tasks.find(*it) !=
                 output.relocate_tasks.end()))
            {
              // Remove it from our set of local tasks
              const ContextID ctx_id =
                  (*it)->get_context()->get_logical_tree_context();
              ContextState& state = context_states[ctx_id];
              legion_assert(state.owned_tasks > 0);
              state.owned_tasks--;
              if (state.active && (state.owned_tasks == 0))
                decrement_active_contexts();
              if ((*it)->is_forward_progress_task())
                decrement_progress_tasks();
              to_trigger.emplace_back(*it);
              it = rqueue.erase(it);
            }
            else
              it++;
          }
          if (rqueue.empty())
          {
            if (map_state.deferral_event.exists())
              map_state.deferral_event = RtEvent::NO_RT_EVENT;
            else
              decrement_active_mappers();
          }
          else if (!stealing_disabled)
          {
            for (SingleTask* task : rqueue)
            {
              if (task->is_stealable())
              {
                mappers_with_stealable_work.emplace_back(map_id);
                break;
              }
            }
          }
          // Remove the queue guard
          map_state.queue_guard = false;
          if (map_state.queue_waiter.exists())
          {
            Runtime::trigger_event(map_state.queue_waiter);
            map_state.queue_waiter = RtUserEvent::NO_RT_USER_EVENT;
          }
        }
        // Now we can trigger our tasks that the mapper selected
        std::map<Processor, std::vector<SingleTask*> > to_send;
        for (SingleTask* task : to_trigger)
        {
          // Mark that this task is no longer outstanding
          task->deactivate_outstanding_task();
          // Update the target processor for this task if necessary
          std::map<const Task*, Processor>::const_iterator finder =
              output.relocate_tasks.find(task);
          if (finder != output.relocate_tasks.end())
          {
            task->set_target_proc(finder->second);
            // See if the target processor is local
            if (!runtime->is_local(finder->second))
            {
              // This is the tricky case, we need to actually send this
              // remotely, which is hard if it is a point task that is
              // owned by a slice task, if it is just a normal indvidual
              // task then we can just ship it remotely immediately
              to_send[finder->second].emplace_back(task);
            }
            else
              task->enqueue_ready_task(true /*use target processor*/);
          }
          else
          {
            TaskOp::TriggerTaskArgs trigger_args(
                task, task->get_context()->did);
            runtime->issue_runtime_meta_task(
                trigger_args, LG_THROUGHPUT_WORK_PRIORITY);
          }
        }
        if (!to_send.empty())
        {
          for (std::pair<const Processor, std::vector<SingleTask*> >& it :
               to_send)
            runtime->send_tasks(it.first, it.second);
        }
      }

      // Advertise any work that we have
      if (!stealing_disabled && !mappers_with_stealable_work.empty())
      {
        for (const MapperID& mapper : mappers_with_stealable_work)
          issue_advertisements(mapper);
      }

      // Finally issue any steal requeusts
      if (!stealing_disabled && !stealing_targets.empty())
        runtime->send_steal_request(stealing_targets, local_proc);
    }

    //--------------------------------------------------------------------------
    void ProcessorManager::issue_advertisements(MapperID map_id)
    //--------------------------------------------------------------------------
    {
      // Create a clone of the processors we want to advertise so that
      // we don't call into the high level runtime holding a lock
      std::set<Processor> failed_waiters;
      MapperManager* mapper = find_mapper(map_id);
      mapper->perform_advertisements(failed_waiters);
      if (!failed_waiters.empty())
        runtime->send_advertisements(failed_waiters, map_id, local_proc);
    }

  }  // namespace Internal
}  // namespace Legion
