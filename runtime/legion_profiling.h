/* Copyright 2013 Stanford University
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

namespace LegionRuntime {
  namespace HighLevel {

    // Used for creating LegionProf Recorder
    enum ProfRecorderKind {
      PROF_CREATE_INDEX_SPACE = 0,
      PROF_DESTROY_INDEX_SPACE = 1,
      PROF_CREATE_INDEX_PARTITION = 2,
      PROF_DESTROY_INDEX_PARTITION = 3,
      PROF_GET_INDEX_PARTITION = 4,
      PROF_GET_INDEX_SUBSPACE = 5,
      PROF_GET_INDEX_DOMAIN = 6,
      PROF_GET_INDEX_PARTITION_COLOR_SPACE = 7,
      PROF_SAFE_CAST = 8,
      PROF_CREATE_FIELD_SPACE = 9,
      PROF_DESTROY_FIELD_SPACE = 10,
      PROF_ALLOCATE_FIELDS = 11,
      PROF_FREE_FIELDS = 12,
      PROF_CREATE_REGION = 13,
      PROF_DESTROY_REGION = 14,
      PROF_DESTROY_PARTITION = 15,
      PROF_GET_LOGICAL_PARTITION = 16,
      PROF_GET_LOGICAL_SUBREGION = 17,
      PROF_MAP_REGION = 18,
      PROF_UNMAP_REGION = 19,
      PROF_TASK_DEP_ANALYSIS = 20,
      PROF_MAP_DEP_ANALYSIS = 21,
      PROF_DEL_DEP_ANALYSIS = 22,
      PROF_SCHEDULER = 23,
      PROF_TASK_MAP = 24,
      PROF_TASK_RUN = 25,
      PROF_TASK_CHILDREN_MAPPED= 26,
      PROF_TASK_FINISH = 27,
    };

    namespace LegionProf {

      // These numbers exactly match the meanings
      // in legion_prof.py
      enum ProfKind {
        BEGIN_INDEX_SPACE_CREATE = 0,
        END_INDEX_SPACE_CREATE = 1,
        BEGIN_INDEX_SPACE_DESTROY = 2,
        END_INDEX_SPACE_DESTROY = 3,
        BEGIN_INDEX_PARTITION_CREATE = 4,
        END_INDEX_PARTITION_CREATE = 5,
        BEGIN_INDEX_PARTITION_DESTROY = 6,
        END_INDEX_PARTITION_DESTROY = 7,
        BEGIN_GET_INDEX_PARTITION = 8,
        END_GET_INDEX_PARTITION = 9,
        BEGIN_GET_INDEX_SUBSPACE = 10,
        END_GET_INDEX_SUBSPACE = 11,
        BEGIN_GET_INDEX_DOMAIN = 12,
        END_GET_INDEX_DOMAIN = 13,
        BEGIN_GET_INDEX_PARTITION_COLOR_SPACE = 14,
        END_GET_INDEX_PARTITION_COLOR_SPACE = 15,
        BEGIN_SAFE_CAST = 16,
        END_SAFE_CAST = 17,
        BEGIN_CREATE_FIELD_SPACE = 18,
        END_CREATE_FIELD_SPACE = 19,
        BEGIN_DESTROY_FIELD_SPACE = 20,
        END_DESTROY_FIELD_SPACE = 21,
        BEGIN_ALLOCATE_FIELDS = 22,
        END_ALLOCATE_FIELDS = 23,
        BEGIN_FREE_FIELDS = 24,
        END_FREE_FIELDS = 25,
        BEGIN_CREATE_REGION = 26,
        END_CREATE_REGION = 27,
        BEGIN_DESTROY_REGION = 28,
        END_DESTROY_REGION = 29,
        BEGIN_DESTROY_PARTITION = 30,
        END_DESTROY_PARTITION = 31,
        BEGIN_GET_LOGICAL_PARTITION = 32,
        END_GET_LOGICAL_PARTITION = 33,
        BEGIN_GET_LOGICAL_SUBREGION = 34,
        END_GET_LOGICAL_SUBREGION = 35,
        BEGIN_MAP_REGION = 36,
        END_MAP_REGION = 37,
        BEGIN_UNMAP_REGION = 38,
        END_UNMAP_REGION = 39,
        BEGIN_TASK_DEP_ANALYSIS = 40,
        END_TASK_DEP_ANALYSIS = 41,
        BEGIN_MAP_DEP_ANALYSIS = 42,
        END_MAP_DEP_ANALYSIS = 43,
        BEGIN_DEL_DEP_ANALYSIS = 44,
        END_DEL_DEP_ANALYSIS = 45,
        BEGIN_SCHEDULER = 46,
        END_SCHEDULER = 47,
        BEGIN_TASK_MAP = 48,
        END_TASK_MAP = 49,
        BEGIN_TASK_RUN = 50,
        END_TASK_RUN = 51,
        BEGIN_TASK_CHILDREN_MAPPED = 52,
        END_TASK_CHILDREN_MAPPED = 53,
        BEGIN_TASK_FINISH = 54,
        END_TASK_FINISH = 55,
        TASK_LAUNCH = 56, // this should always be last
      }; 

      struct ProfilingEvent {
      public:
        ProfilingEvent(unsigned k, unsigned tid, unsigned uid, const DomainPoint &p, unsigned long long t)
          : kind(k), task_id(tid), unique_id(uid), point(p), time(t) { }
      public:
        unsigned kind;
        unsigned task_id;
        unsigned unique_id;
        DomainPoint point;
        unsigned long long time; // absolute time in micro-seconds 
      };

      struct MemoryEvent {
      public:
        MemoryEvent(unsigned iid, unsigned uid, unsigned mem, unsigned r, 
            unsigned bf, const std::map<unsigned,size_t> &fields, unsigned long long t)
          : creation(true), inst_id(iid), unique_id(uid), memory(mem), 
            redop(r), blocking_factor(bf), time(t), field_infos(fields) { }
        MemoryEvent(unsigned uid, unsigned long long t)
          : creation(false), unique_id(uid), time(t) { }
      public:
        bool creation;
        unsigned inst_id;
        unsigned unique_id;
        unsigned memory;
        unsigned redop;
        size_t blocking_factor;
        unsigned long long time;
        std::map<unsigned,size_t> field_infos;
      };

      struct TaskInstance {
      public:
        TaskInstance(void) { }
        TaskInstance(unsigned tid, unsigned uid, const DomainPoint &p)
          : task_id(tid), unique_id(uid), point(p) { }
      public:
        unsigned task_id;
        unsigned unique_id;
        DomainPoint point;
      };

      struct ProcessorProfiler {
      public:
        ProcessorProfiler(void)
          : proc(Processor::NO_PROC), utility(false), init_time(0) { }
        ProcessorProfiler(Processor p, bool util, Processor::Kind k) 
          : proc(p), utility(util), kind(k),
            init_time(TimeStamp::get_current_time_in_micros()) { }
      public:
        void add_event(const ProfilingEvent &event) { proc_events.push_back(event); }
        void add_event(const MemoryEvent &event) { mem_events.push_back(event); }
        void add_subtask(unsigned suid, const TaskInstance &inst) { sub_tasks[suid] = inst; }
      public:
        Processor proc;
        bool utility;
        Processor::Kind kind;
        unsigned long long init_time;
        std::vector<ProfilingEvent> proc_events;
        std::vector<MemoryEvent> mem_events;
        std::map<unsigned,TaskInstance> sub_tasks;
      };

      extern Logger::Category log_prof;
      // Profiler table indexed by processor id
      extern ProcessorProfiler *legion_prof_table;

      static inline ProcessorProfiler& get_profiler(Processor proc)
      {
        return legion_prof_table[(proc.id & 0xffff)];
      }

      static inline void register_task_variant(unsigned task_id, bool leaf, const char *name)
      {
        log_prof(LEVEL_INFO,"Prof Task Variant %d %d %s", task_id, leaf, name);
      }

      static inline void initialize_processor(Processor proc, bool util, Processor::Kind kind)
      {
        get_profiler(proc) = ProcessorProfiler(proc, util, kind);
      }

      static inline void initialize_memory(Memory mem, Memory::Kind kind)
      {
        log_prof(LEVEL_INFO,"Prof Memory %x %d", mem.id, kind);
      }

      static inline void finalize_processor(Processor proc)
      {
        ProcessorProfiler &prof = get_profiler(proc);
        log_prof(LEVEL_INFO,"Prof Processor %x %d %d", proc.id, prof.utility, prof.kind);
        for (unsigned idx = 0; idx < prof.proc_events.size(); idx++)
        {
          ProfilingEvent &event = prof.proc_events[idx]; 
          // Probably shouldn't role over, if something did then
          // we may need to change our assumptions
          assert(event.time >= prof.init_time);
          switch (event.kind)
          {
            case BEGIN_SCHEDULER:
            case END_SCHEDULER:    
              {
                log_prof(LEVEL_INFO,"Prof Scheduler %x %d %lld", proc.id, event.kind, (event.time-prof.init_time));
                break;
              }
            default:
              {
                log_prof(LEVEL_INFO,"Prof Task Event %x %d %d %d %lld %d %d %d %d",
                    proc.id, event.kind, event.task_id, event.unique_id, (event.time-prof.init_time),
                    event.point.get_dim(), event.point.point_data[0], event.point.point_data[1],
                    event.point.point_data[2]);
                break;
              }
          }
        }
        for (unsigned idx = 0; idx < prof.mem_events.size(); idx++)
        {
          MemoryEvent &event = prof.mem_events[idx];
          assert(event.time >= prof.init_time);
          if (event.creation)
          {
            // First log the instance information
            log_prof(LEVEL_INFO,"Prof Create Instance %x %d %x %d %ld %lld", 
                event.inst_id, event.unique_id, event.memory, event.redop, 
                event.blocking_factor, (event.time - prof.init_time));
            // Then log the creation of the fields
            for (std::map<unsigned,size_t>::const_iterator it = event.field_infos.begin();
                  it != event.field_infos.end(); it++)
            {
              log_prof(LEVEL_INFO,"Prof Instance Field %d %d %ld", event.unique_id, it->first, it->second);
            }
          }
          else
          {
            // Log the instance destruction
            log_prof(LEVEL_INFO,"Prof Destroy Instance %d %lld", event.unique_id, (event.time - prof.init_time));
          }
        }
        for (std::map<unsigned,TaskInstance>::const_iterator it = prof.sub_tasks.begin();
              it != prof.sub_tasks.end(); it++)
        {
          log_prof(LEVEL_INFO,"Prof Subtask %d %d %d %d %d %d %d",
              it->first, it->second.task_id, it->second.unique_id, it->second.point.get_dim(),
              it->second.point.point_data[0], it->second.point.point_data[1],
              it->second.point.point_data[2]);
        }
      }

      static inline void register_task_begin_run(unsigned task_id, unsigned unique_id, const DomainPoint &point)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_event(ProfilingEvent(BEGIN_TASK_RUN,task_id,unique_id,point,time));
      }

      static inline void register_task_end_run(unsigned task_id, unsigned unique_id, const DomainPoint &point)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_event(ProfilingEvent(END_TASK_RUN,task_id,unique_id,point,time));
      }

      static inline void register_task_launch(unsigned task_id, unsigned unique_id, const DomainPoint &point)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_event(ProfilingEvent(TASK_LAUNCH,task_id,unique_id,point,time));
      }

      static inline void register_sub_task(unsigned task_id, unsigned unique_id, const DomainPoint &point, unsigned sub_unique_id)
      {
        // Kind of tricky here, we actually just need the sub-task relationship, so we'll
        // pass the sub-task's unique id where the timing information normally goes.
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_subtask(sub_unique_id,TaskInstance(task_id,unique_id,point));
      }

      static inline void register_task_begin_children_mapped(unsigned task_id, unsigned unique_id, const DomainPoint &point)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_event(ProfilingEvent(BEGIN_TASK_CHILDREN_MAPPED,task_id,unique_id,point,time));
      }

      static inline void register_task_end_children_mapped(unsigned task_id, unsigned unique_id, const DomainPoint &point)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_event(ProfilingEvent(END_TASK_CHILDREN_MAPPED,task_id,unique_id,point,time));
      }

      static inline void register_task_begin_finish(unsigned task_id, unsigned unique_id, const DomainPoint &point)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_event(ProfilingEvent(BEGIN_TASK_FINISH,task_id,unique_id,point,time));
      }

      static inline void register_task_end_finish(unsigned task_id, unsigned unique_id, const DomainPoint &point)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_event(ProfilingEvent(END_TASK_FINISH,task_id,unique_id,point,time));
      }

      static inline void register_instance_creation(unsigned inst_id, unsigned unique_id,
        unsigned memory, unsigned redop, size_t blocking_factor, const std::map<unsigned,size_t> &fields)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_event(MemoryEvent(inst_id, unique_id, memory, redop, blocking_factor, fields, time));
      }

      static inline void register_instance_deletion(unsigned unique_id)
      {
        unsigned long long time = TimeStamp::get_current_time_in_micros();
        Processor proc = Machine::get_executing_processor();
        get_profiler(proc).add_event(MemoryEvent(unique_id, time));
      }

      template<int REC>
      class Recorder {
      public:
        Recorder(unsigned task_id, unsigned unique_id, const DomainPoint &point)
          : tid(task_id), uid(unique_id), p(point)
        {
          unsigned long long time = TimeStamp::get_current_time_in_micros();
          Processor proc = Machine::get_executing_processor();
          get_profiler(proc).add_event(ProfilingEvent(REC*2,tid,uid,p,time));
        }
        ~Recorder(void)
        {
          unsigned long long time = TimeStamp::get_current_time_in_micros();
          Processor proc = Machine::get_executing_processor();
          get_profiler(proc).add_event(ProfilingEvent(REC*2+1,tid,uid,p,time));
        }
      private:
        unsigned tid;
        unsigned uid;
        const DomainPoint &p;
      };

    };
  };
};

#endif // __LEGION_PROFILING_H__

