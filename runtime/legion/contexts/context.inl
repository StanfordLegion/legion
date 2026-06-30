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

// Included from context.h - do not include this directly

// Useful for IDEs
#include "legion/contexts/context.h"

namespace Legion {
  namespace Internal {

    //--------------------------------------------------------------------------
    inline bool TaskContext::begin_runtime_call(
        RuntimeCallKind kind, Provenance* provenance)
    //--------------------------------------------------------------------------
    {
      if (overhead_profiler != nullptr)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff =
            current - overhead_profiler->previous_profiling_time;
        overhead_profiler->application_time += diff;
        overhead_profiler->previous_profiling_time = current;
        overhead_profiler->inside_runtime_call = true;
      }
      return ((runtime->profiler != nullptr) || (overhead_profiler != nullptr));
    }

    //--------------------------------------------------------------------------
    inline void TaskContext::end_runtime_call(
        RuntimeCallKind kind, Provenance* provenance, unsigned long long start,
        unsigned long long stop)
    //--------------------------------------------------------------------------
    {
      if (overhead_profiler != nullptr)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff =
            current - overhead_profiler->previous_profiling_time;
        overhead_profiler->runtime_time += diff;
        overhead_profiler->previous_profiling_time = current;
        overhead_profiler->inside_runtime_call = false;
      }
      if (implicit_profiler != nullptr)
        implicit_profiler->record_runtime_call(kind, start, stop);
    }

    //--------------------------------------------------------------------------
    inline void TaskContext::begin_wait(LgEvent event, bool from_application)
    //--------------------------------------------------------------------------
    {
      if (overhead_profiler != nullptr)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff =
            current - overhead_profiler->previous_profiling_time;
        if (overhead_profiler->inside_runtime_call)
          overhead_profiler->runtime_time += diff;
        else
          overhead_profiler->application_time += diff;
        overhead_profiler->previous_profiling_time = current;
      }
      if (implicit_task_profiler != nullptr)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        implicit_task_profiler->waits.emplace_back(
            LegionProfInstance::WaitInfo{current, current, current, event});
      }
    }

    //--------------------------------------------------------------------------
    inline void TaskContext::end_wait(LgEvent event, bool from_application)
    //--------------------------------------------------------------------------
    {
      if (overhead_profiler != nullptr)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        const long long diff =
            current - overhead_profiler->previous_profiling_time;
        overhead_profiler->wait_time += diff;
        overhead_profiler->previous_profiling_time = current;
      }
      if (implicit_task_profiler != nullptr)
      {
        const long long current = Realm::Clock::current_time_in_nanoseconds();
        legion_assert(!implicit_task_profiler->waits.empty());
        LegionProfInstance::WaitInfo& info =
            implicit_task_profiler->waits.back();
        legion_assert(info.wait_event == event);
        // Assume that implicit tasks resume as soon as the event is triggered
        info.wait_ready = current;
        info.wait_end = current;
      }
    }

    //--------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const TaskContext& ctx)
    //--------------------------------------------------------------------------
    {
      os << ctx.get_task_name() << "(UID: " << ctx.get_unique_id() << ")";
      return os;
    }

  }  // namespace Internal
}  // namespace Legion
