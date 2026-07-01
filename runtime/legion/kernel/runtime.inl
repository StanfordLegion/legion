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

// Included from runtime.h - do not include this directly

// Useful for IDEs
#include "legion/kernel/runtime.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    template<typename T>
    inline RtEvent Runtime::issue_runtime_meta_task(
        const LgTaskArgs<T>& args, LgPriority priority, RtEvent precondition,
        Processor target)
    //--------------------------------------------------------------------------
    {
      // If this is not a task directly related to shutdown or is a message,
      // to a remote node then increment the number of outstanding tasks
      if (T::TASK_ID < LG_BEGIN_SHUTDOWN_TASK_IDS)
        increment_total_outstanding_tasks(args.lg_task_id, true /*meta*/);
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
      outstanding_counts[T::TASK_ID].fetch_add(1);
#endif
      if (!target.exists())
      {
        // If we don't have a processor to explicitly target, figure
        // out which of our utility processors to use
        target = utility_group;
      }
      legion_assert(target.exists());
      if (profiler != nullptr)
      {
        Realm::ProfilingRequestSet requests;
        profiler->add_meta_request(
            requests, T::TASK_ID, args.unique_op_id, precondition);
#ifdef LEGION_SEPARATE_META_TASKS
        return RtEvent(target.spawn(
            LG_TASK_ID + T::TASK_ID, &args, sizeof(T), requests, precondition,
            priority));
#else
        return RtEvent(target.spawn(
            LG_TASK_ID, &args, sizeof(T), requests, precondition, priority));
#endif
      }
      else
#ifdef LEGION_SEPARATE_META_TASKS
        return RtEvent(target.spawn(
            LG_TASK_ID + T::TASK_ID, &args, sizeof(T), precondition, priority));
#else
        return RtEvent(
            target.spawn(LG_TASK_ID, &args, sizeof(T), precondition, priority));
#endif
    }

    //--------------------------------------------------------------------------
    template<typename T>
    inline RtEvent Runtime::issue_application_processor_task(
        const LgTaskArgs<T>& args, LgPriority priority, const Processor target,
        RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      static_assert(T::IS_APPLICATION_TASK);
      static_assert(
          T::TASK_ID < LG_BEGIN_SHUTDOWN_TASK_IDS,
          "Shutdown tasks should never be run directly on application procs");
      // If this is not a task directly related to shutdown or is a message,
      // to a remote node then increment the number of outstanding tasks
      legion_assert(target.exists());
      legion_assert(target.kind() != Processor::UTIL_PROC);
      increment_total_outstanding_tasks(args.lg_task_id, true /*meta*/);
#ifdef LEGION_DEBUG_SHUTDOWN_HANG
      outstanding_counts[T::TASK_ID].fetch_add(1);
#endif
      if (profiler != nullptr)
      {
        Realm::ProfilingRequestSet requests;
        profiler->add_meta_request(
            requests, T::TASK_ID, args.unique_op_id, precondition);
#ifdef LEGION_SEPARATE_META_TASKS
        return RtEvent(target.spawn(
            LG_APP_PROC_TASK_ID + T::TASK_ID, &args, sizeof(T), requests,
            precondition, priority));
#else
        return RtEvent(target.spawn(
            LG_APP_PROC_TASK_ID, &args, sizeof(T), requests, precondition,
            priority));
#endif
      }
      else
#ifdef LEGION_SEPARATE_META_TASKS
        return RtEvent(target.spawn(
            LG_APP_PROC_TASK_ID + T::TASK_ID, &args, sizeof(T), precondition,
            priority));
#else
        return RtEvent(target.spawn(
            LG_APP_PROC_TASK_ID, &args, sizeof(T), precondition, priority));
#endif
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::merge_events(
        const TraceInfo* info, ApEvent e1, ApEvent e2)
    //--------------------------------------------------------------------------
    {
      ApEvent result(Realm::Event::merge_events(e1, e2));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == e1) || (result == e2)))
        rename_event(result);
      LegionSpy::log_event_dependence(e1, result);
      LegionSpy::log_event_dependence(e2, result);
      if ((implicit_profiler != nullptr) && result.exists())
      {
        const LgEvent preconditions[2] = {e1, e2};
        implicit_profiler->record_event_merger(result, preconditions, 2);
      }
      // Always do tracing after profiling
      if ((info != nullptr) && info->recording)
      {
        const ApEvent rhs[2] = {e1, e2};
        info->record_merge_events(result, rhs, 2);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::merge_events(
        const TraceInfo* info, ApEvent e1, ApEvent e2, ApEvent e3)
    //--------------------------------------------------------------------------
    {
      ApEvent result(Realm::Event::merge_events(e1, e2, e3));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == e1) || (result == e2) ||
           (result == e3)))
        rename_event(result);
      LegionSpy::log_event_dependence(e1, result);
      LegionSpy::log_event_dependence(e2, result);
      LegionSpy::log_event_dependence(e3, result);
      if ((implicit_profiler != nullptr) && result.exists())
      {
        const LgEvent preconditions[3] = {e1, e2, e3};
        implicit_profiler->record_event_merger(result, preconditions, 3);
      }
      // Always do tracing after profiling
      if ((info != nullptr) && info->recording)
      {
        const ApEvent rhs[3] = {e1, e2, e3};
        info->record_merge_events(result, rhs, 3);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename A>
    /*static*/ inline ApEvent Runtime::merge_events(
        const TraceInfo* info, const std::set<ApEvent, A>& events)
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level <= LIGHT_SPY_LOGGING)
      {
        if (events.empty())
        {
          // Still need to do this for tracing because of merge filter code
          if ((info != nullptr) && info->recording)
          {
            ApEvent result;
            info->record_merge_events(result, &result, 0);
            return result;
          }
          else
            return ApEvent::NO_AP_EVENT;
        }
        if (events.size() == 1)
        {
          // Still need to do this for tracing because of merge filter code
          if ((info != nullptr) && info->recording)
          {
            ApEvent event = *(events.begin());
            ApEvent result = event;
            info->record_merge_events(result, &event, 1);
            return result;
          }
          else
            return *(events.begin());
        }
      }
      // Fuck C++
      const std::set<ApEvent, A>* legion_events = &events;
      const std::set<Realm::Event>* realm_events;
      static_assert(sizeof(legion_events) == sizeof(realm_events));
      static_assert(sizeof(ApEvent) == sizeof(Realm::Event));
      std::memcpy(&realm_events, &legion_events, sizeof(legion_events));
      ApEvent result(Realm::Event::merge_events(*realm_events));
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        if (!result.exists() || (events.find(result) != events.end()))
          rename_event(result);
        for (const ApEvent& event : events)
          LegionSpy::log_event_dependence(event, result);
      }
      if ((implicit_profiler != nullptr) && result.exists())
      {
        static_assert(sizeof(ApEvent) == sizeof(LgEvent));
        const local::vector<ApEvent> preconditions(
            events.begin(), events.end());
        implicit_profiler->record_event_merger(
            result, &preconditions.front(), preconditions.size());
      }
      // Always do tracing after profiling
      if ((info != nullptr) && info->recording)
      {
        const local::vector<ApEvent> preconditions(
            events.begin(), events.end());
        info->record_merge_events(
            result, &preconditions.front(), preconditions.size());
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename A>
    /*static*/ inline ApEvent Runtime::merge_events(
        const TraceInfo* info, const std::vector<ApEvent, A>& events)
    //--------------------------------------------------------------------------
    {
      if (spy_logging_level <= LIGHT_SPY_LOGGING)
      {
        if (events.empty())
        {
          // Still need to do this for tracing because of merge filter code
          if ((info != nullptr) && info->recording)
          {
            ApEvent result;
            info->record_merge_events(result, &result, 0);
            return result;
          }
          else
            return ApEvent::NO_AP_EVENT;
        }
        if (events.size() == 1)
        {
          // Still need to do this for tracing because of merge filter code
          if ((info != nullptr) && info->recording)
          {
            ApEvent result = events.front();
            info->record_merge_events(result, &events.front(), 1);
            return result;
          }
          else
            return events.front();
        }
      }
      // Fuck C++
      const std::vector<ApEvent>* legion_events = &events;
      const std::vector<Realm::Event>* realm_events;
      static_assert(sizeof(legion_events) == sizeof(realm_events));
      static_assert(sizeof(ApEvent) == sizeof(Realm::Event));
      std::memcpy(&realm_events, &legion_events, sizeof(legion_events));
      ApEvent result(Realm::Event::merge_events(*realm_events));
      if (spy_logging_level > LIGHT_SPY_LOGGING)
      {
        if (result.exists())
        {
          // Check to make sure it isn't a rename
          for (unsigned idx = 0; idx < events.size(); idx++)
          {
            if (events[idx] != result)
              continue;
            rename_event(result);
            break;
          }
        }
        else
          rename_event(result);
        for (const ApEvent& event : events)
          LegionSpy::log_event_dependence(event, result);
      }
      if ((implicit_profiler != nullptr) && result.exists())
      {
        static_assert(sizeof(ApEvent) == sizeof(LgEvent));
        implicit_profiler->record_event_merger(
            result, &events.front(), events.size());
      }
      // Always do tracing after profiling
      if ((info != nullptr) && info->recording)
        info->record_merge_events(result, &events.front(), events.size());
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::merge_events(RtEvent e1, RtEvent e2)
    //--------------------------------------------------------------------------
    {
      // No logging for runtime operations currently
      RtEvent result(Realm::Event::merge_events(e1, e2));
      if ((implicit_profiler != nullptr) && result.exists())
      {
        const LgEvent preconditions[2] = {e1, e2};
        implicit_profiler->record_event_merger(result, preconditions, 2);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::merge_events(
        RtEvent e1, RtEvent e2, RtEvent e3)
    //--------------------------------------------------------------------------
    {
      // No logging for runtime operations currently
      const RtEvent result(Realm::Event::merge_events(e1, e2, e3));
      if ((implicit_profiler != nullptr) && result.exists())
      {
        const LgEvent preconditions[3] = {e1, e2, e3};
        implicit_profiler->record_event_merger(result, preconditions, 3);
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename A>
    /*static*/ inline RtEvent Runtime::merge_events(
        const std::set<RtEvent, A>& events)
    //--------------------------------------------------------------------------
    {
      if (events.empty())
        return RtEvent::NO_RT_EVENT;
      if (events.size() == 1)
        return *(events.begin());
      // Fuck C++
      const std::set<RtEvent, A>* legion_events = &events;
      const std::set<Realm::Event>* realm_events;
      static_assert(sizeof(legion_events) == sizeof(realm_events));
      static_assert(sizeof(RtEvent) == sizeof(Realm::Event));
      std::memcpy(&realm_events, &legion_events, sizeof(legion_events));
      // No logging for runtime operations currently
      const RtEvent result(Realm::Event::merge_events(*realm_events));
      if ((implicit_profiler != nullptr) && result.exists())
      {
        static_assert(sizeof(RtEvent) == sizeof(LgEvent));
        const local::vector<RtEvent> preconditions(
            events.begin(), events.end());
        implicit_profiler->record_event_merger(
            result, &preconditions.front(), preconditions.size());
      }
      return result;
    }

    //--------------------------------------------------------------------------
    template<typename A>
    /*static*/ inline RtEvent Runtime::merge_events(
        const std::vector<RtEvent, A>& events)
    //--------------------------------------------------------------------------
    {
      if (events.empty())
        return RtEvent::NO_RT_EVENT;
      if (events.size() == 1)
        return events.front();
      // Fuck C++
      const std::vector<RtEvent, A>* legion_events = &events;
      const std::vector<Realm::Event>* realm_events;
      static_assert(sizeof(legion_events) == sizeof(realm_events));
      static_assert(sizeof(RtEvent) == sizeof(Realm::Event));
      std::memcpy(&realm_events, &legion_events, sizeof(legion_events));
      // No logging for runtime operations currently
      const RtEvent result(Realm::Event::merge_events(*realm_events));
      if ((implicit_profiler != nullptr) && result.exists())
      {
        static_assert(sizeof(RtEvent) == sizeof(LgEvent));
        implicit_profiler->record_event_merger(
            result, &events.front(), events.size());
      }
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApUserEvent Runtime::create_ap_user_event(
        const TraceInfo* info)
    //--------------------------------------------------------------------------
    {
      ApUserEvent result;
      if ((info == nullptr) || !info->recording)
      {
        result = ApUserEvent(Realm::UserEvent::create_user_event());
        LegionSpy::log_ap_user_event(result);
      }
      else
        info->record_create_ap_user_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::trigger_event_untraced(
        ApUserEvent to_trigger, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      // Record trigger event timing first since it might be expensive
      // to actually propagate the triggered event
      if (implicit_profiler != nullptr)
        implicit_profiler->record_event_trigger(to_trigger, precondition);
      Realm::UserEvent copy = to_trigger;
      copy.trigger(precondition);
      LegionSpy::log_ap_user_event_trigger(to_trigger);
      if (precondition.exists())
        LegionSpy::log_event_dependence(precondition, to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::trigger_event(
        ApUserEvent to_trigger, ApEvent precondition, const TraceInfo& info,
        std::set<RtEvent>& applied)
    //--------------------------------------------------------------------------
    {
      // Record trigger event timing first since it might be expensive
      // to actually propagate the triggered event
      if (implicit_profiler != nullptr)
        implicit_profiler->record_event_trigger(to_trigger, precondition);
      Realm::UserEvent copy = to_trigger;
      copy.trigger(precondition);
      LegionSpy::log_ap_user_event_trigger(to_trigger);
      if (precondition.exists())
        LegionSpy::log_event_dependence(precondition, to_trigger);
      if (info.recording)
        info.record_trigger_event(to_trigger, precondition, applied);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::poison_event(ApUserEvent to_poison)
    //--------------------------------------------------------------------------
    {
      // Record poison event timing first since it might be expensive
      // to actually propagate the poison
      if (implicit_profiler != nullptr)
        implicit_profiler->record_event_poison(to_poison);
      Realm::UserEvent copy = to_poison;
      copy.cancel();
      // This counts as triggering
      LegionSpy::log_ap_user_event_trigger(to_poison);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtUserEvent Runtime::create_rt_user_event(void)
    //--------------------------------------------------------------------------
    {
      RtUserEvent result(Realm::UserEvent::create_user_event());
      LegionSpy::log_rt_user_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::trigger_event(
        RtUserEvent to_trigger, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      // Record trigger event timing first since it might be expensive
      // to actually propagate the triggered event
      if (implicit_profiler != nullptr)
        implicit_profiler->record_event_trigger(to_trigger, precondition);
      Realm::UserEvent copy = to_trigger;
      copy.trigger(precondition);
      LegionSpy::log_rt_user_event_trigger(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::poison_event(RtUserEvent to_poison)
    //--------------------------------------------------------------------------
    {
      // Record poison event timing first since it might be expensive
      // to actually propagate the poison
      if (implicit_profiler != nullptr)
        implicit_profiler->record_event_poison(to_poison);
      Realm::UserEvent copy = to_poison;
      copy.cancel();
      // This counts as triggering
      LegionSpy::log_rt_user_event_trigger(to_poison);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline PredUserEvent Runtime::create_pred_event(void)
    //--------------------------------------------------------------------------
    {
      PredUserEvent result(Realm::UserEvent::create_user_event());
      LegionSpy::log_pred_event(result);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::trigger_event(PredUserEvent to_trigger)
    //--------------------------------------------------------------------------
    {
      // Record trigger event timing first since it might be expensive
      // to actually propagate the triggered event
      if (implicit_profiler != nullptr)
        implicit_profiler->record_event_trigger(
            to_trigger, LgEvent::NO_LG_EVENT);
      Realm::UserEvent copy = to_trigger;
      copy.trigger();
      LegionSpy::log_pred_event_trigger(to_trigger);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::poison_event(PredUserEvent to_poison)
    //--------------------------------------------------------------------------
    {
      // Record poison event timing first since it might be expensive
      // to actually propagate the poison
      if (implicit_profiler != nullptr)
        implicit_profiler->record_event_poison(to_poison);
      Realm::UserEvent copy = to_poison;
      copy.cancel();
      // This counts as triggering
      LegionSpy::log_pred_event_trigger(to_poison);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline PredEvent Runtime::merge_events(
        const TraceInfo* info, PredEvent e1, PredEvent e2)
    //--------------------------------------------------------------------------
    {
      legion_assert(e1.exists());
      legion_assert(e2.exists());
      PredEvent result(Realm::Event::merge_events(e1, e2));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) &&
          (!result.exists() || (result == e1) || (result == e2)))
        rename_event(result);
      LegionSpy::log_event_dependence(e1, result);
      LegionSpy::log_event_dependence(e2, result);
      if ((implicit_profiler != nullptr) && result.exists())
      {
        const LgEvent preconditions[2] = {e1, e2};
        implicit_profiler->record_event_merger(result, preconditions, 2);
      }
      // Always do tracing after profiling
      if ((info != nullptr) && info->recording)
        info->record_merge_events(result, e1, e2);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::ignorefaults(ApEvent e)
    //--------------------------------------------------------------------------
    {
      ApEvent result(Realm::Event::ignorefaults(e));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) && !result.exists())
        rename_event(result);
      LegionSpy::log_event_dependence(ApEvent(e), result);
      if ((implicit_profiler != nullptr) && result.exists() && (result != e))
        implicit_profiler->record_event_trigger(result, e);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::protect_event(ApEvent to_protect)
    //--------------------------------------------------------------------------
    {
      if (to_protect.exists())
      {
        const RtEvent result(Realm::Event::ignorefaults(to_protect));
        if ((implicit_profiler != nullptr) && result.exists() &&
            (result.id != to_protect.id))
          implicit_profiler->record_event_trigger(result, to_protect);
        return result;
      }
      else
        return RtEvent::NO_RT_EVENT;
    }

    //--------------------------------------------------------------------------
    template<typename A>
    /*static*/ inline RtEvent Runtime::protect_merge_events(
        const std::set<ApEvent, A>& events)
    //--------------------------------------------------------------------------
    {
      const std::set<ApEvent, A>* ptr = &events;
      const std::set<Realm::Event>* realm_events = nullptr;
      static_assert(sizeof(realm_events) == sizeof(ptr));
      memcpy(&realm_events, &ptr, sizeof(realm_events));
      RtEvent result(Realm::Event::merge_events_ignorefaults(*realm_events));
      if ((implicit_profiler != nullptr) && result.exists())
      {
        static_assert(sizeof(ApEvent) == sizeof(LgEvent));
        const std::vector<ApEvent> preconditions(events.begin(), events.end());
        implicit_profiler->record_event_merger(
            result, &preconditions.front(), preconditions.size());
      }
      return result;
    }

    //--------------------------------------------------------------------------
    inline void Runtime::phase_barrier_arrive(
        const PhaseBarrier& bar, unsigned count, ApEvent precondition,
        const void* reduce_value, size_t reduce_value_size)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar.phase_barrier;
      // Record barrier arrivals before we actually do them since it
      // might be expensive to propagate the effects
      if ((profiler != nullptr) && !profiler->no_critical_paths)
      {
        if (profiler->all_critical_arrivals)
        {
          if (implicit_profiler != nullptr)
            implicit_profiler->record_barrier_arrival(
                bar.phase_barrier, precondition);
          copy.arrive(count, precondition, reduce_value, reduce_value_size);
        }
        else
        {
          legion_assert(reduce_value == nullptr);
          // We're computing the critical path through the graph so
          // need to record when when this arrival triggers so we can
          // feed it into the reduction for the barrier
          const Realm::Event pre =
              precondition.exists() ? Realm::Event::ignorefaults(precondition) :
                                      precondition;
          if (!pre.exists() || pre.has_triggered())
          {
            const ArrivalInfo info(precondition);
            copy.arrive(count, precondition, &info, sizeof(info));
          }
          else  // Have the profiler profile and do the arrival
            profiler->profile_barrier_arrival(copy, count, precondition, pre);
        }
      }
      else
        copy.arrive(count, precondition, reduce_value, reduce_value_size);
      if (precondition.exists())
        LegionSpy::log_event_dependence(precondition, bar.phase_barrier);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApBarrier Runtime::get_previous_phase(
        const PhaseBarrier& bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar.phase_barrier;
      return ApBarrier(copy.get_previous_phase());
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::alter_arrival_count(
        PhaseBarrier& bar, int delta)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar.phase_barrier;
      bar.phase_barrier = ApBarrier(copy.alter_arrival_count(delta));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::advance_barrier(PhaseBarrier& bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar.phase_barrier;
      bar.phase_barrier = ApBarrier(copy.advance_barrier());
    }

    //--------------------------------------------------------------------------
    inline ApBarrier Runtime::create_ap_barrier(size_t arrivals)
    //--------------------------------------------------------------------------
    {
      if ((profiler == nullptr) || profiler->no_critical_paths ||
          profiler->all_critical_arrivals)
        return ApBarrier(Realm::Barrier::create_barrier(arrivals));
      else
        return ApBarrier(Realm::Barrier::create_barrier(
            arrivals, BarrierArrivalReduction::REDOP,
            &BarrierArrivalReduction::identity,
            sizeof(BarrierArrivalReduction::identity)));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApBarrier Runtime::get_previous_phase(
        const ApBarrier& bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      return ApBarrier(copy.get_previous_phase());
    }

    //--------------------------------------------------------------------------
    inline void Runtime::phase_barrier_arrive(
        const ApBarrier& bar, unsigned count, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      // Record barrier arrivals before we actually do them since it
      // might be expensive to propagate the effects
      if ((profiler != nullptr) && !profiler->no_critical_paths)
      {
        if (profiler->all_critical_arrivals)
        {
          if (implicit_profiler != nullptr)
            implicit_profiler->record_barrier_arrival(bar, precondition);
          copy.arrive(count, precondition);
        }
        else
        {
          // We're computing the critical path through the graph so
          // need to record when when this arrival triggers so we can
          // feed it into the reduction for the barrier
          const Realm::Event pre =
              precondition.exists() ? Realm::Event::ignorefaults(precondition) :
                                      precondition;
          if (!pre.exists() || pre.has_triggered())
          {
            const ArrivalInfo info(precondition);
            copy.arrive(count, precondition, &info, sizeof(info));
          }
          else  // Have the profiler profile and do the arrival
            profiler->profile_barrier_arrival(bar, count, precondition, pre);
        }
      }
      else
        copy.arrive(count, precondition);
      if (precondition.exists())
        LegionSpy::log_event_dependence(precondition, bar);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::advance_barrier(ApBarrier& bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      bar = ApBarrier(copy.advance_barrier());
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool Runtime::get_barrier_result(
        ApBarrier bar, void* result, size_t result_size)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      return copy.get_result(result, result_size);
    }

    //--------------------------------------------------------------------------
    inline RtBarrier Runtime::create_rt_barrier(size_t arrivals)
    //--------------------------------------------------------------------------
    {
      if ((profiler == nullptr) || profiler->no_critical_paths ||
          profiler->all_critical_arrivals)
        return RtBarrier(Realm::Barrier::create_barrier(arrivals));
      else
        return RtBarrier(Realm::Barrier::create_barrier(
            arrivals, BarrierArrivalReduction::REDOP,
            &BarrierArrivalReduction::identity,
            sizeof(BarrierArrivalReduction::identity)));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtBarrier Runtime::get_previous_phase(const RtBarrier& b)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = b;
      return RtBarrier(copy.get_previous_phase());
    }

#ifdef LEGION_DEBUG_COLLECTIVES
    //--------------------------------------------------------------------------
    inline void Runtime::phase_barrier_arrive(
        const RtBarrier& bar, unsigned count, RtEvent precondition,
        const void* value, size_t size)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      // Record barrier arrivals before we actually do them since it
      // might be expensive to propagate the effects
      if ((profiler != nullptr) && !profiler->no_critical_paths)
      {
        legion_assert(profiler->all_critical_arrivals);
        if (implicit_profiler != nullptr)
          implicit_profiler->record_barrier_arrival(bar, precondition);
      }
      copy.arrive(count, precondition, value, size);
    }
#else
    //--------------------------------------------------------------------------
    inline void Runtime::phase_barrier_arrive(
        const RtBarrier& bar, unsigned count, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      // Record barrier arrivals before we actually do them since it
      // might be expensive to propagate the effects
      if ((profiler != nullptr) && !profiler->no_critical_paths)
      {
        if (profiler->all_critical_arrivals)
        {
          if (implicit_profiler != nullptr)
            implicit_profiler->record_barrier_arrival(bar, precondition);
          copy.arrive(count, precondition);
        }
        else
        {
          // We're computing the critical path through the graph so
          // need to record when when this arrival triggers so we can
          // feed it into the reduction for the barrier
          if (!precondition.exists() || precondition.has_triggered())
          {
            const ArrivalInfo info(precondition);
            copy.arrive(count, precondition, &info, sizeof(info));
          }
          else  // Have the profiler profile and do the arrival
            profiler->profile_barrier_arrival(
                bar, count, precondition, precondition);
        }
      }
      else
        copy.arrive(count, precondition);
    }
#endif

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::advance_barrier(RtBarrier& bar)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      bar = RtBarrier(copy.advance_barrier());
    }

    //--------------------------------------------------------------------------
    /*static*/ inline bool Runtime::get_barrier_result(
        RtBarrier bar, void* result, size_t result_size)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = bar;
      return copy.get_result(result, result_size);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::alter_arrival_count(RtBarrier& b, int delta)
    //--------------------------------------------------------------------------
    {
      Realm::Barrier copy = b;
      b = RtBarrier(copy.alter_arrival_count(delta));
    }

    //--------------------------------------------------------------------------
    /*static*/ inline ApEvent Runtime::acquire_ap_reservation(
        Reservation r, bool exclusive, ApEvent precondition)
    //--------------------------------------------------------------------------
    {
      ApEvent result(r.acquire(exclusive ? 0 : 1, exclusive, precondition));
      if ((spy_logging_level > LIGHT_SPY_LOGGING) && precondition.exists() &&
          !result.exists())
        rename_event(result);
      LegionSpy::log_reservation_acquire(r, precondition, result);
      // Result can be the same as precondition if precondition is poisoned
      if ((implicit_profiler != nullptr) && result.exists() &&
          (result != precondition))
        implicit_profiler->record_reservation_acquire(r, result, precondition);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline RtEvent Runtime::acquire_rt_reservation(
        Reservation r, bool exclusive, RtEvent precondition)
    //--------------------------------------------------------------------------
    {
      RtEvent result(r.acquire(exclusive ? 0 : 1, exclusive, precondition));
      // Result can be the same as precondition if precondition is poisoned
      if ((implicit_profiler != nullptr) && result.exists() &&
          (result != precondition))
        implicit_profiler->record_reservation_acquire(r, result, precondition);
      return result;
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::release_reservation(
        Reservation r, LgEvent precondition)
    //--------------------------------------------------------------------------
    {
      r.release(precondition);
    }

    //--------------------------------------------------------------------------
    /*static*/ inline void Runtime::rename_event(LgEvent& to_rename)
    //--------------------------------------------------------------------------
    {
      const Realm::UserEvent renamed = Realm::UserEvent::create_user_event();
      renamed.trigger(to_rename);
      to_rename = LgEvent(renamed);
    }

  }  // namespace Internal
}  // namespace Legion
