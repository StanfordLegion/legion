/* Copyright 2015 Stanford University, NVIDIA Corporation
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
#include "realm/profiling.h"

#include <cassert>
#include <deque>
#include <algorithm>

namespace LegionRuntime {
  namespace HighLevel {

    class LegionProfInstance {
    public:
      struct TaskKind {
      public:
        Processor::TaskFuncID task_id;
        const char *task_name;
      };
      struct TaskVariant {
      public:
        Processor::TaskFuncID func_id;
        const char *variant_name;
      };
      struct OperationInstance {
      public:
        UniqueID op_id;
        unsigned op_kind;
      };
      struct MultiTask {
      public:
        UniqueID op_id;
        Processor::TaskFuncID task_id;
      };
      struct TaskInfo {
      public:
        UniqueID task_id;
        Processor::TaskFuncID func_id;
        Processor proc;
        unsigned long long create, ready, start, stop;
      };
      struct MetaInfo {
      public:
        UniqueID op_id;
        unsigned hlr_id;
        Processor proc;
        unsigned long long create, ready, start, stop;
      };
      struct CopyInfo {
      public:
        UniqueID op_id;
        Memory source, target;
        unsigned long long create, ready, start, stop;
      };
      struct FillInfo {
      public:
        UniqueID op_id;
        Memory target;
        unsigned long long create, ready, start, stop;
      };
      struct InstInfo {
      public:
        UniqueID op_id; 
        PhysicalInstance inst;
        Memory mem;
        size_t total_bytes;
        unsigned long long create, destroy;
      };
    public:
      LegionProfInstance(LegionProfiler *owner);
      LegionProfInstance(const LegionProfInstance &rhs);
      ~LegionProfInstance(void);
    public:
      LegionProfInstance& operator=(const LegionProfInstance &rhs);
    public:
      void register_task_kind(Processor::TaskFuncID kind, const char *name);
      void register_task_variant(const char *variant_name,
                                 const TaskVariantCollection::Variant &variant);
      void register_operation(Operation *op);
      void register_multi_task(Operation *op, Processor::TaskFuncID kind);
    public:
      void process_task(size_t id, UniqueID op_id, 
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage);
      void process_meta(size_t id, UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage);
      void process_copy(UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationMemoryUsage *usage);
      void process_fill(UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationMemoryUsage *usage);
      void process_inst(UniqueID op_id,
                  Realm::ProfilingMeasurements::InstanceTimeline *timeline,
                  Realm::ProfilingMeasurements::InstanceMemoryUsage *usage);
    public:
      void dump_state(void);
    private:
      LegionProfiler *const owner;
      std::deque<TaskKind>          task_kinds;
      std::deque<TaskVariant>       task_variants;
      std::deque<OperationInstance> operation_instances;
      std::deque<MultiTask>         multi_tasks;
    private:
      std::deque<TaskInfo> task_infos;
      std::deque<MetaInfo> meta_infos;
      std::deque<CopyInfo> copy_infos;
      std::deque<FillInfo> fill_infos;
      std::deque<InstInfo> inst_infos;
    };

    class LegionProfiler {
    public:
      enum ProfilingKind {
        LEGION_PROF_TASK,
        LEGION_PROF_META,
        LEGION_PROF_COPY,
        LEGION_PROF_FILL,
        LEGION_PROF_INST,
      };
      struct ProfilingInfo {
      public:
        ProfilingInfo(ProfilingKind k)
          : kind(k) { }
      public:
        ProfilingKind kind;
        size_t id;
        UniqueID op_id;
      };
    public:
      // Statically known information passed through the constructor
      // so that it can be deduplicated
      LegionProfiler(Processor target_proc, const Machine &machine,
                     unsigned num_meta_tasks,
                     const char *const *const meta_task_descriptions,
                     unsigned num_operation_kinds,
                     const char *const *const operation_kind_descriptions);
      LegionProfiler(const LegionProfiler &rhs);
      ~LegionProfiler(void);
    public:
      LegionProfiler& operator=(const LegionProfiler &rhs);
    public:
      // Dynamically created things must be registered at runtime
      // Tasks
      void register_task_kind(Processor::TaskFuncID task_id,
                              const char *task_name);
      void register_task_variant(const char *variant_name,
                                 const TaskVariantCollection::Variant &variant);
      // Operations
      void register_operation(Operation *op);
      void register_multi_task(Operation *op, Processor::TaskFuncID task_id);
    public:
      void add_task_request(Realm::ProfilingRequestSet &requests, 
                            Processor::TaskFuncID tid, SingleTask *task);
      void add_meta_request(Realm::ProfilingRequestSet &requests,
                            HLRTaskID tid, Operation *op);
      void add_copy_request(Realm::ProfilingRequestSet &requests, 
                            Operation *op);
      void add_fill_request(Realm::ProfilingRequestSet &requests,
                            Operation *op);
      void add_inst_request(Realm::ProfilingRequestSet &requests,
                            Operation *op);
    public:
      // Process low-level runtime profiling results
      void process_results(Processor p, const void *buffer, size_t size);
    public:
      // Dump all the results
      void finalize(void);
    public:
      const Processor target_proc;
    private:
      LegionProfInstance **const instances;
    };

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
      PROF_BEGIN_COPY = 22,
      PROF_END_COPY = 23,
      // User-defined profiling events
      PROF_BEGIN_USER_EVENT = 100,
      PROF_END_USER_EVENT = 101,
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
          : proc(Processor::NO_PROC), utility(false), dumped(0),
            init_time(0) { }
        ProcessorProfiler(Processor p, bool util, Processor::Kind k)
          : proc(p), utility(util), kind(k), dumped(0),
            init_time(Realm::Clock::current_time_in_microseconds()) { }
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

      struct CopyProfiler {
      public:
        CopyProfiler(void) { }
      public:
        inline void add_event(const ProfilingEvent &event)
        { proc_events.push_back(event); }
      private:
        // no copy constructor or assignment
        CopyProfiler(const CopyProfiler& copy_from)
        { assert(false); }
        CopyProfiler& operator=(const CopyProfiler& copy_from)
        { assert(false); return *this; }
      public:
        std::deque<ProfilingEvent> proc_events;
      };

      extern Logger::Category log_prof;
      // Profiler table indexed by processor id
      extern unsigned long long legion_prof_init_time;
      extern ProcessorProfiler *legion_prof_table;
      extern pthread_key_t copy_profiler_key;
      extern pthread_mutex_t copy_profiler_mutex;
      extern std::list<CopyProfiler*> copy_prof_list;
      extern int copy_profiler_dumped;
      // Indicator for when profiling is enabled and disabled
      extern bool profiling_enabled;

      static inline void init_timestamp()
      {
        legion_prof_init_time = Realm::Clock::current_time_in_microseconds();
      }

      static inline ProcessorProfiler& get_profiler(Processor proc)
      {
        return legion_prof_table[proc.local_id()];
      }

      static inline CopyProfiler& get_copy_profiler()
      {
        CopyProfiler* copy_prof =
          reinterpret_cast<CopyProfiler*>(
              pthread_getspecific(copy_profiler_key));
        if (!copy_prof)
        {
          copy_prof = new CopyProfiler();
          pthread_setspecific(copy_profiler_key, copy_prof);
          pthread_mutex_lock(&copy_profiler_mutex);
          copy_prof_list.push_back(copy_prof);
          pthread_mutex_unlock(&copy_profiler_mutex);
        }
        return *copy_prof;
      }

      static inline void register_task_variant(unsigned task_id,
                                               const char *name)
      {
        if (profiling_enabled)
          log_prof.info("Prof Task Variant %u %s", task_id, name);
      }

      static inline void initialize_processor(Processor proc,
                                              bool util,
                                              Processor::Kind kind)
      {
        ProcessorProfiler &p = get_profiler(proc);
        new (&p) ProcessorProfiler(proc, util, kind);
      }

      static inline void initialize_memory(Memory mem, Memory::Kind kind)
      {
        if (profiling_enabled)
          log_prof.info("Prof Memory " IDFMT " %u", mem.id, kind);
      }

      static inline void finalize_processor(Processor proc)
      {
        ProcessorProfiler &prof = get_profiler(proc);
        int perform_dump = __sync_fetch_and_add(&prof.dumped, 1);
        // Someone else has already dumped this processor
        if (perform_dump > 0)
          return;
        if (profiling_enabled)
          log_prof.info("Prof Processor " IDFMT " %u %u",
                    proc.id, prof.utility, prof.kind);
        for (unsigned idx = 0; idx < prof.tasks.size(); idx++)
        {
          const TaskInstance &inst = prof.tasks[idx];
          log_prof.info("Prof Unique Task " IDFMT " %llu %u %u %u %u %u",
              proc.id, inst.unique_id, inst.task_id,
              inst.point.get_dim(), inst.point.point_data[0],
              inst.point.point_data[1], inst.point.point_data[2]);
        }
        prof.tasks.clear();
        for (unsigned idx = 0; idx < prof.mappings.size(); idx++)
        {
          const OpInstance &inst = prof.mappings[idx];
          log_prof.info("Prof Unique Map " IDFMT " %llu %llu",
              proc.id, inst.unique_id, inst.parent_id);
        }
        prof.mappings.clear();
        for (unsigned idx = 0; idx < prof.closes.size(); idx++)
        {
          const OpInstance &inst = prof.closes[idx];
          log_prof.info("Prof Unique Close " IDFMT " %llu %llu",
              proc.id, inst.unique_id, inst.parent_id);
        }
        prof.closes.clear();
        for (unsigned idx = 0; idx < prof.copies.size(); idx++)
        {
          const OpInstance &inst = prof.copies[idx];
          log_prof.info("Prof Unique Copy " IDFMT " %llu %llu",
              proc.id, inst.unique_id, inst.parent_id);
        }
        prof.copies.clear();
        for (unsigned idx = 0; idx < prof.proc_events.size(); idx++)
        {
          ProfilingEvent &event = prof.proc_events[idx];
          // Probably shouldn't role over, if something did then
          // we may need to change our assumptions
          assert(event.time >= legion_prof_init_time);
          log_prof.info("Prof Event " IDFMT " %u %llu %llu",
            proc.id, event.kind, event.unique_id,
            (event.time-legion_prof_init_time));
        }
        prof.proc_events.clear();
        for (unsigned idx = 0; idx < prof.mem_events.size(); idx++)
        {
          MemoryEvent &event = prof.mem_events[idx];
          assert(event.time >= legion_prof_init_time);
          if (event.creation)
          {
            // First log the instance information
            log_prof.info("Prof Create Instance %u %u %u %ld %llu",
                event.inst_id, event.memory, event.redop,
                event.blocking_factor, (event.time - legion_prof_init_time));
            // Then log the creation of the fields
            for (std::map<unsigned,size_t>::const_iterator it =
                  event.field_infos.begin(); it !=
                  event.field_infos.end(); it++)
            {
              log_prof.info("Prof Instance Field %u %u %ld",
                  event.inst_id, it->first, it->second);
            }
          }
          else
          {
            // Log the instance destruction
            log_prof.info("Prof Destroy Instance %u %llu",
                event.inst_id , (event.time - legion_prof_init_time));
          }
        }
        prof.mem_events.clear();
      }

      static bool compare_events(ProfilingEvent* ev1, ProfilingEvent* ev2)
      {
        if (ev1->kind < ev2->kind) return true;
        else if (ev1->kind > ev2->kind) return false;
        else return ev1->time < ev2->time;
      }

      static inline void finalize_copy_profiler()
      {
        legion_lowlevel_id_t last_proc_id = 0;
        for (unsigned idx = 0; idx < (MAX_NUM_PROCS+1); idx++)
        {
          Processor proc = legion_prof_table[idx].proc;
          if (proc.exists() && last_proc_id < proc.id)
            last_proc_id = proc.id;
        }
        last_proc_id += 1;

        int perform_dump = __sync_fetch_and_add(&copy_profiler_dumped, 1);
        if (perform_dump > 0) return;

        if (profiling_enabled)
        {
          typedef std::map<UniqueID, std::deque<ProfilingEvent*> > event_map_t;
          event_map_t copy_events;

          for (std::list<CopyProfiler*>::iterator it = copy_prof_list.begin();
              it != copy_prof_list.end(); ++it)
          {
            CopyProfiler* copy_prof = *it;
            for (unsigned idx = 0; idx < copy_prof->proc_events.size(); idx++)
            {
              ProfilingEvent &event = copy_prof->proc_events[idx];
              copy_events[event.unique_id].push_back(&event);
            }
          }

          log_prof.info("Prof Processor " IDFMT " 1 3", last_proc_id);

          for (event_map_t::iterator it = copy_events.begin();
               it != copy_events.end(); ++it)
          {
            std::deque<ProfilingEvent*>& events = it->second;
            assert(events.size() % 2 == 0);
            std::sort(events.begin(), events.end(), compare_events);
            int num_copies = events.size() / 2;
            for (int idx = 0; idx < num_copies; ++idx)
            {
              ProfilingEvent &begin_event = *events[idx];
              ProfilingEvent &end_event = *events[idx + num_copies];
              log_prof.info("Prof Event " IDFMT " %u 0 %llu",
                  last_proc_id, begin_event.kind,
                  begin_event.time - legion_prof_init_time);
              log_prof.info("Prof Event " IDFMT " %u 0 %llu",
                  last_proc_id, end_event.kind,
                  end_event.time - legion_prof_init_time);
            }
          }

          for (std::list<CopyProfiler*>::iterator it = copy_prof_list.begin();
              it != copy_prof_list.end(); ++it)
          {
            delete *it;
          }
        }
      }

      static inline void register_copy_event(UniqueID uid, ProfKind kind)
      {
        if (profiling_enabled)
        {
          CopyProfiler &copy_prof = get_copy_profiler();
          unsigned long long time = Realm::Clock::current_time_in_microseconds();
          copy_prof.add_event(ProfilingEvent(kind, uid, time));
        }
      }

      static inline void register_event(UniqueID uid, ProfKind kind)
      {
        if (profiling_enabled)
        {
          unsigned long long time = Realm::Clock::current_time_in_microseconds();
          Processor proc = Processor::get_executing_processor();
          get_profiler(proc).add_event(ProfilingEvent(kind, uid, time));
        }
      }

      static inline void register_task(unsigned tid, UniqueID uid,
                                       const DomainPoint &point)
      {
        if (profiling_enabled)
        {
          Processor proc = Processor::get_executing_processor();
          get_profiler(proc).add_task(TaskInstance(tid, uid, point));
        }
      }

      static inline void register_map(UniqueID uid, UniqueID pid)
      {
        if (profiling_enabled)
        {
          Processor proc = Processor::get_executing_processor();
          get_profiler(proc).add_map(OpInstance(uid, pid));
        }
      }

      static inline void register_close(UniqueID uid, UniqueID pid)
      {
        if (profiling_enabled)
        {
          Processor proc = Processor::get_executing_processor();
          get_profiler(proc).add_close(OpInstance(uid, pid));
        }
      }

      static inline void register_copy(UniqueID uid, UniqueID pid)
      {
        if (profiling_enabled)
        {
          Processor proc = Processor::get_executing_processor();
          get_profiler(proc).add_copy(OpInstance(uid, pid));
        }
      }

      static inline void register_instance_creation(unsigned inst_id,
        unsigned memory, unsigned redop,
        size_t blocking_factor, const std::map<unsigned,size_t> &fields)
      {
        if (profiling_enabled)
        {
          unsigned long long time = Realm::Clock::current_time_in_microseconds();
          Processor proc = Processor::get_executing_processor();
          get_profiler(proc).add_event(MemoryEvent(inst_id, memory,
                                        redop, blocking_factor, fields, time));
        }
      }

      static inline void register_instance_deletion(unsigned inst_id)
      {
        if (profiling_enabled)
        {
          unsigned long long time = Realm::Clock::current_time_in_microseconds();
          Processor proc = Processor::get_executing_processor();
          get_profiler(proc).add_event(MemoryEvent(inst_id, time));
        }
      }

      static inline void register_userevent(UniqueID uid, const char* name)
      {
        if (profiling_enabled)
        {
          Processor proc = Processor::get_executing_processor();
          log_prof.info("Prof User Event " IDFMT " %llu %s",
              proc.id, uid, name);
        }
      }

      static inline void begin_userevent(UniqueID uid)
      {
        register_event(uid, PROF_BEGIN_USER_EVENT);
      }

      static inline void end_userevent(UniqueID uid)
      {
        register_event(uid, PROF_END_USER_EVENT);
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

