/* Copyright 2014 Stanford University
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


#ifndef __LEGION_PROFILING_H__
#define __LEGION_PROFILING_H__

#include "lowlevel.h"
#include "utilities.h"
#include "legion_types.h"
#include "legion_utilities.h"

#include <cassert>
#include <deque>

namespace LegionRuntime {
  namespace HighLevel {

    // Used for creating LegionProf Recorder
    enum ProfKind {
      PROF_BEGIN_DEP_ANALYSIS = 0,
      PROF_END_DEP_ANALYSIS = 1,
      PROF_BEGIN_PREMAP_ANALYSIS = 2,
      PROF_END_PREMAP_ANALYSIS = 3,
      PROF_BEGIN_MAP_ANALYSIS = 4,
      PROF_END_MAP_ANALYSIS = 5,
      PROF_BEGIN_EXECUTION = 6,
      PROF_END_EXECUTION = 7,
      PROF_BEGIN_WAIT = 8,
      PROF_END_WAIT = 9,
      PROF_BEGIN_SCHEDULER = 10,
      PROF_END_SCHEDULER = 11,
      // Non begin-end pairs
      PROF_COMPLETE = 12,
      PROF_LAUNCH = 13,
      // Other begin-end pairs
      PROF_BEGIN_POST = 14,
      PROF_END_POST = 15,
      PROF_BEGIN_TRIGGER = 16,
      PROF_END_TRIGGER = 17,
      PROF_BEGIN_GC = 18,
      PROF_END_GC = 19,
      PROF_BEGIN_MESSAGE = 20,
      PROF_END_MESSAGE = 21,
    };
 
    namespace LegionProf {

      struct ProfilingEvent {
      public:
        ProfilingEvent(unsigned k, UniqueID uid, unsigned long long t)
          : kind(k), unique_id(uid), time(t) { }
      public:
        unsigned kind;
        UniqueID unique_id;
        unsigned long long time; // absolute time in micro-seconds 
      };

      struct MemoryEvent {
      public:
        MemoryEvent(unsigned iid, unsigned mem, unsigned r, 
                    unsigned bf, const std::map<unsigned,size_t> &fields, 
                    unsigned long long t)
          : creation(true), inst_id(iid), memory(mem), 
            redop(r), blocking_factor(bf), time(t), field_infos(fields) { }
        MemoryEvent(unsigned iid, unsigned long long t)
          : creation(false), inst_id(iid), time(t) { }
      public:
        bool creation;
        unsigned inst_id;
        unsigned memory;
        unsigned redop;
        size_t blocking_factor;
        unsigned long long time;
        std::map<unsigned,size_t> field_infos;
      };

      struct TaskInstance {
      public:
        TaskInstance(void) { }
        TaskInstance(unsigned tid, UniqueID uid, const DomainPoint &p)
          : task_id(tid), unique_id(uid), point(p) { }
      public:
        unsigned task_id;
        UniqueID unique_id;
        DomainPoint point;
      };

      struct OpInstance {
      public:
        OpInstance(void) { }
        OpInstance(UniqueID uid, UniqueID pid)
          : unique_id(uid), parent_id(pid) { }
      public:
        UniqueID unique_id;
        UniqueID parent_id;
      };

      struct ProcessorProfiler {
      public:
        ProcessorProfiler(void)
          : proc(Processor::NO_PROC), utility(false), init_time(0) { }
        ProcessorProfiler(Processor p, bool util, Processor::Kind k) 
          : proc(p), utility(util), kind(k), dumped(0),
            init_time(TimeStamp::get_current_time_in_micros()) { }
      public:
        inline void add_event(const ProfilingEvent &event) 
                          { proc_events.push_back(event); }
        inline void add_event(const MemoryEvent &event) 
                          { mem_events.push_back(event); }
        inline void add_task(const TaskInstance &inst)
                          { tasks.push_back(inst); }
        inline void add_map(const OpInstance &inst)
                          { mappings.push_back(inst); }
        inline void add_close(const OpInstance &inst)
                          { closes.push_back(inst); }
        inline void add_copy(const OpInstance &inst)
                          { copies.push_back(inst); }
      private:
	// no copy constructor or assignment
	ProcessorProfiler(const ProcessorProfiler& copy_from) 
        { assert(false); }
	ProcessorProfiler& operator=(const ProcessorProfiler& copy_from) 
        { assert(false); return *this; }
      public:
        Processor proc;
        bool utility;
        Processor::Kind kind;
        int dumped;
        unsigned long long init_time;
        std::deque<ProfilingEvent> proc_events;
        std::deque<MemoryEvent> mem_events;
        std::deque<TaskInstance> tasks;
        std::deque<OpInstance> mappings;
        std::deque<OpInstance> closes;
        std::deque<OpInstance> copies;
      };

      extern Logger::Category log_prof;
      // Profiler table indexed by processor id
      extern ProcessorProfiler *legion_prof_table;
      // Indicator for when profiling is enabled and disabled
      extern bool profiling_enabled;

      static inline ProcessorProfiler& get_profiler(Processor proc)
      {
        return legion_prof_table[proc.local_id()];
      }

      static inline void register_task_variant(unsigned task_id, 
                                               const char *name)
      {
        if (profiling_enabled)
          log_prof(LEVEL_INFO,"Prof Task Variant %u %s", task_id, name);
      }

      static inline void initialize_processor(Processor proc, 
                                              bool util, 
                                              Processor::Kind kind)
      {
	ProcessorProfiler &p = get_profiler(proc);
	p.proc = proc;
	p.utility = util;
	p.kind = kind;
	p.init_time = TimeStamp::get_current_time_in_micros();
      }

      static inline void initialize_memory(Memory mem, Memory::Kind kind)
      {
        if (profiling_enabled)
          log_prof(LEVEL_INFO,"Prof Memory " IDFMT " %u", mem.id, kind);
      }

      static inline void finalize_processor(Processor proc)
      {
        ProcessorProfiler &prof = get_profiler(proc);
        int perform_dump = __sync_fetch_and_add(&prof.dumped, 1);
        // Someone else has already dumped this processor
        if (perform_dump > 0)
          return;
        if (profiling_enabled)
          log_prof(LEVEL_INFO,"Prof Processor " IDFMT " %u %u", 
                    proc.id, prof.utility, prof.kind);
        for (unsigned idx = 0; idx < prof.tasks.size(); idx++)
        {
          const TaskInstance &inst = prof.tasks[idx];
          log_prof(LEVEL_INFO,"Prof Unique Task " IDFMT " %llu %u %u %u %u %u",
              proc.id, inst.unique_id, inst.task_id, 
              inst.point.get_dim(), inst.point.point_data[0],
              inst.point.point_data[1], inst.point.point_data[2]);
        }
        prof.tasks.clear();
        for (unsigned idx = 0; idx < prof.mappings.size(); idx++)
        {
          const OpInstance &inst = prof.mappings[idx];
          log_prof(LEVEL_INFO,"Prof Unique Map " IDFMT " %llu %llu",
              proc.id, inst.unique_id, inst.parent_id);
        }
        prof.mappings.clear();
        for (unsigned idx = 0; idx < prof.closes.size(); idx++)
        {
          const OpInstance &inst = prof.closes[idx];
          log_prof(LEVEL_INFO,"Prof Unique Close " IDFMT " %llu %llu",
              proc.id, inst.unique_id, inst.parent_id);
        }
        prof.closes.clear();
        for (unsigned idx = 0; idx < prof.copies.size(); idx++)
        {
          const OpInstance &inst = prof.copies[idx];
          log_prof(LEVEL_INFO,"Prof Unique Copy " IDFMT " %llu %llu",
              proc.id, inst.unique_id, inst.parent_id);
        }
        prof.copies.clear();
        for (unsigned idx = 0; idx < prof.proc_events.size(); idx++)
        {
          ProfilingEvent &event = prof.proc_events[idx]; 
          // Probably shouldn't role over, if something did then
          // we may need to change our assumptions
          assert(event.time >= prof.init_time);
          log_prof(LEVEL_INFO, "Prof Event " IDFMT " %u %llu %llu", 
            proc.id, event.kind, event.unique_id, (event.time-prof.init_time));
        }
        prof.proc_events.clear();
        for (unsigned idx = 0; idx < prof.mem_events.size(); idx++)
        {
          MemoryEvent &event = prof.mem_events[idx];
          assert(event.time >= prof.init_time);
          if (event.creation)
          {
            // First log the instance information
            log_prof(LEVEL_INFO,"Prof Create Instance %u %u %u %ld %llu", 
                event.inst_id, event.memory, event.redop, 
                event.blocking_factor, (event.time - prof.init_time));
            // Then log the creation of the fields
            for (std::map<unsigned,size_t>::const_iterator it = 
                  event.field_infos.begin(); it != 
                  event.field_infos.end(); it++)
            {
              log_prof(LEVEL_INFO,"Prof Instance Field %u %u %ld", 
                  event.inst_id, it->first, it->second);
            }
          }
          else
          {
            // Log the instance destruction
            log_prof(LEVEL_INFO,"Prof Destroy Instance %u %llu", 
                event.inst_id , (event.time - prof.init_time));
          }
        }
        prof.mem_events.clear();
      }

      static inline void register_event(UniqueID uid, ProfKind kind)
      {
        if (profiling_enabled)
        {
          unsigned long long time = TimeStamp::get_current_time_in_micros();
          Processor proc = Machine::get_executing_processor();
          get_profiler(proc).add_event(ProfilingEvent(kind, uid, time));
        }
      }

      static inline void register_task(unsigned tid, UniqueID uid, 
                                       const DomainPoint &point)
      {
        if (profiling_enabled)
        {
          Processor proc = Machine::get_executing_processor();
          get_profiler(proc).add_task(TaskInstance(tid, uid, point));
        }
      }

      static inline void register_map(UniqueID uid, UniqueID pid)
      {
        if (profiling_enabled)
        {
          Processor proc = Machine::get_executing_processor();
          get_profiler(proc).add_map(OpInstance(uid, pid));
        }
      }

      static inline void register_close(UniqueID uid, UniqueID pid)
      {
        if (profiling_enabled)
        {
          Processor proc = Machine::get_executing_processor();
          get_profiler(proc).add_close(OpInstance(uid, pid));
        }
      }

      static inline void register_copy(UniqueID uid, UniqueID pid)
      {
        if (profiling_enabled)
        {
          Processor proc = Machine::get_executing_processor();
          get_profiler(proc).add_copy(OpInstance(uid, pid));
        }
      }

      static inline void register_instance_creation(unsigned inst_id, 
        unsigned memory, unsigned redop, 
        size_t blocking_factor, const std::map<unsigned,size_t> &fields)
      {
        if (profiling_enabled)
        {
          unsigned long long time = TimeStamp::get_current_time_in_micros();
          Processor proc = Machine::get_executing_processor();
          get_profiler(proc).add_event(MemoryEvent(inst_id, memory, 
                                        redop, blocking_factor, fields, time));
        }
      }

      static inline void register_instance_deletion(unsigned inst_id)
      {
        if (profiling_enabled)
        {
          unsigned long long time = TimeStamp::get_current_time_in_micros();
          Processor proc = Machine::get_executing_processor();
          get_profiler(proc).add_event(MemoryEvent(inst_id, time));
        }
      }

      static inline void enable_profiling(void)
      {
        profiling_enabled = true;        
      }

      static inline void disable_profiling(void)
      {
        profiling_enabled = false;
      }

      static inline void dump_profiling(void)
      {
        for (unsigned idx = 0; idx < (MAX_NUM_PROCS+1); idx++)
        {
          Processor proc = legion_prof_table[idx].proc;
          if (proc.exists())
            finalize_processor(proc);
        }
      }

    };
  };
};

#endif // __LEGION_PROFILING_H__

