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

#include "legion/kernel/metatask.h"
#include "legion/api/types.h"

#ifndef __LEGION_PROCESSOR_MANAGER_H__
#define __LEGION_PROCESSOR_MANAGER_H__

namespace Legion {
  namespace Internal {

    /**
     * \class ProcessorManager
     * This class manages all the state for a single processor
     * within a given instance of the Internal runtime.  It keeps
     * queues for each of the different stages that operations
     * undergo and also tracks when the scheduling task needs
     * to be run for a processor.
     */
    class ProcessorManager {
    public:
      struct SchedulerArgs : public LgTaskArgs<SchedulerArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_SCHEDULER_ID;
      public:
        SchedulerArgs(void) = default;
        SchedulerArgs(ProcessorManager* m)
          : LgTaskArgs<SchedulerArgs>(true, true), manager(m)
        { }
        void execute(void) const;
      public:
        ProcessorManager* manager;
      };
      struct DeferMapperSchedulerArgs
        : public LgTaskArgs<DeferMapperSchedulerArgs> {
      public:
        static constexpr LgTaskID TASK_ID = LG_DEFER_MAPPER_SCHEDULER_TASK_ID;
      public:
        DeferMapperSchedulerArgs(void) = default;
        DeferMapperSchedulerArgs(
            ProcessorManager* proxy, MapperID mid, RtEvent defer)
          : LgTaskArgs<DeferMapperSchedulerArgs>(true, true), proxy_this(proxy),
            map_id(mid), deferral_event(defer)
        { }
        void execute(void) const;
      public:
        ProcessorManager* proxy_this;
        MapperID map_id;
        RtEvent deferral_event;
      };
      struct MapperMessage {
      public:
        MapperMessage(void)
          : target(Processor::NO_PROC), message(nullptr), length(0), radix(0)
        { }
        MapperMessage(Processor t, void* mes, size_t l)
          : target(t), message(mes), length(l), radix(-1)
        { }
        MapperMessage(void* mes, size_t l, int r)
          : target(Processor::NO_PROC), message(mes), length(l), radix(r)
        { }
      public:
        Processor target;
        void* message;
        size_t length;
        int radix;
      };
    public:
      ProcessorManager(
          Processor proc, Processor::Kind proc_kind, unsigned default_mappers,
          bool no_steal, bool replay);
      ProcessorManager(const ProcessorManager& rhs) = delete;
      ~ProcessorManager(void);
    public:
      ProcessorManager& operator=(const ProcessorManager& rhs) = delete;
    public:
      void prepare_for_shutdown(void);
    public:
      void add_mapper(
          MapperID mid, MapperManager* m, bool check, bool skip_replay = false);
      void replace_default_mapper(MapperManager* m);
      MapperManager* find_mapper(MapperID mid) const;
      bool has_non_default_mapper(void) const;
    public:
      void perform_scheduling(void);
      void launch_task_scheduler(void);
      void notify_deferred_mapper(MapperID map_id, RtEvent deferred_event);
    public:
      void activate_context(InnerContext* context);
      void deactivate_context(InnerContext* context);
      void update_max_context_count(unsigned max_contexts);
    public:
      void process_steal_request(
          Processor thief, const std::vector<MapperID>& thieves);
      void process_advertisement(Processor advertiser, MapperID mid);
    public:
      void add_to_ready_queue(SingleTask* task);
    public:
      inline bool is_visible_memory(Memory memory) const
      {
        return (visible_memories.find(memory) != visible_memories.end());
      }
      void find_visible_memories(std::set<Memory>& visible) const;
      Memory find_best_visible_memory(Memory::Kind kind) const;
    public:
      // This method will perform the computation needed to order concurrent
      // index space task launches and trigger the ready event with the
      // precondition event once it is safe to do so
      void order_concurrent_task_launch(
          SingleTask* task, ApEvent precondition, ApUserEvent ready,
          VariantID vid);
      // Once the concurrent index space task launch has performed its max
      // all-reduce of the lamport clocks across all the points then it needs
      // to report the resulting clock back to the processor
      void finalize_concurrent_task_order(
          SingleTask* task, uint64_t lamport, bool poisoned);
      // Report when we are done executing a concurrent index space task and
      // therefore it is safe to beging the next one on this processor
      void end_concurrent_task(void);
    protected:
      void start_next_concurrent_task(void);
    protected:
      void perform_mapping_operations(void);
      void issue_advertisements(MapperID mid);
    protected:
      void increment_active_contexts(void);
      void decrement_active_contexts(void);
    protected:
      void increment_active_mappers(void);
      void decrement_active_mappers(void);
    protected:
      void increment_progress_tasks(void);
      void decrement_progress_tasks(void);
    public:
      // Immutable state
      const Processor local_proc;
      const Processor::Kind proc_kind;
      // Is stealing disabled
      const bool stealing_disabled;
      // are we doing replay execution
      const bool replay_execution;
    protected:
      // Local queue state
      mutable LocalLock local_queue_lock;
      unsigned next_local_index;
    protected:
      // Scheduling state
      mutable LocalLock queue_lock;
      bool task_scheduler_enabled;
      bool outstanding_task_scheduler;
      unsigned total_active_contexts;
      unsigned total_active_mappers;
      // Progress tasks are tasks that have to be mapped in order
      // to guarantee forward progress of the program, these include
      // slices from dependent index space task launches, slices from
      // collectively mapped index task launches, and concurrent
      // index space task launches. If we have a progress task then
      // we need to keep calling select_tasks_to_map until the mapper
      // maps these tasks regardless of whether their context is
      // active or not to avoid hanging waiting for them to map
      unsigned total_progress_tasks;
      struct ContextState {
      public:
        ContextState(void) : owned_tasks(0), active(false) { }
      public:
        unsigned owned_tasks;
        bool active;
      };
      std::vector<ContextState> context_states;
    protected:
      // Mapper objects
      std::map<MapperID, MapperManager*> mappers;
      // For each mapper something to track its state
      struct MapperState {
      public:
        MapperState(void) : queue_guard(false), queue_dirty(false) { }
      public:
        std::list<SingleTask*> ready_queue;
        RtEvent deferral_event;
        RtUserEvent queue_waiter;
        bool queue_guard;
        // If new tasks were added to the ready queue while the queue flag was
        // set
        bool queue_dirty;
      };
      // State for each mapper for scheduling purposes
      std::map<MapperID, MapperState> mapper_states;
      // Lock for accessing mappers
      mutable LocalLock mapper_lock;
      // The set of visible memories from this processor
      std::map<Memory, size_t /*bandwidth affinity*/> visible_memories;
    protected:
      // Data structures to help with the management of concurrent index
      // space task launches. We track a lamport clock for helping to
      // order all concurrent index space task launches that overlap on
      // the same kind of processors.
      mutable LocalLock concurrent_lock;
      struct ConcurrentState {
      public:
        ConcurrentState(uint64_t clock, ApEvent pre, ApUserEvent r)
          : lamport_clock(clock), precondition(pre), ready(r), max(false)
        { }
      public:
        uint64_t lamport_clock;
        ApEvent precondition;
        ApUserEvent ready;
        bool max;  // whether the lamport clock is the max all-reduce or not
      };
      std::map<SingleTask*, ConcurrentState> concurrent_tasks;
      uint64_t concurrent_lamport_clock;
      uint32_t ready_concurrent_tasks;
      bool outstanding_concurrent_task;
    };

  }  // namespace Internal
}  // namespace Legion

#endif  // __LEGION_PROCESSOR_MANAGER_H__
