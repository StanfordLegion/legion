/* Copyright 2017 Stanford University, NVIDIA Corporation
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

#include "realm.h"
#include "utilities.h"
#include "legion_types.h"
#include "legion_utilities.h"
#include "realm/profiling.h"

#include <cassert>
#include <deque>
#include <algorithm>

#define LEGION_PROF_SELF_PROFILE

#ifdef DETAILED_LEGION_PROF
#define DETAILED_PROFILER(runtime, call) DetailedProfiler(runtime, call)
#else
#define DETAILED_PROFILER(runtime, call) // Nothing
#endif

namespace Legion {
  namespace Internal { 

    class LegionProfMarker {
    public:
      LegionProfMarker(const char* name);
      ~LegionProfMarker();
      void mark_stop();
    private:
      const char* name;
      bool stopped;
      Processor proc;
      unsigned long long start, stop;
    };

    class LegionProfInstance {
    public:
      struct TaskKind {
      public:
        TaskID task_id;
        const char *task_name;
        bool overwrite;
      };
      struct TaskVariant {
      public:
        TaskID task_id;
        VariantID variant_id;
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
        TaskID task_id;
      };
      struct SliceOwner {
      public:
        UniqueID parent_id;
        UniqueID op_id;
      };
      struct WaitInfo {
      public:
        unsigned long long wait_start, wait_ready, wait_end;
      };
      struct TaskInfo {
      public:
        UniqueID op_id;
        VariantID variant_id;
        Processor proc;
        unsigned long long create, ready, start, stop;
        std::deque<WaitInfo> wait_intervals;
      };
      struct MetaInfo {
      public:
        UniqueID op_id;
        unsigned lg_id;
        Processor proc;
        unsigned long long create, ready, start, stop;
        std::deque<WaitInfo> wait_intervals;
      };
      struct CopyInfo {
      public:
        UniqueID op_id;
        Memory source, target;
        unsigned long long size;
        unsigned long long create, ready, start, stop;
      };
      struct FillInfo {
      public:
        UniqueID op_id;
        Memory target;
        unsigned long long create, ready, start, stop;
      };
      struct InstCreateInfo {
      public:
	UniqueID op_id;
        PhysicalInstance inst;
	unsigned long long create; // time of HLR creation request
      };
      struct InstUsageInfo {
      public:
	UniqueID op_id;
        PhysicalInstance inst;
        Memory mem;
        size_t total_bytes;
      };
      struct InstTimelineInfo {
      public:
	UniqueID op_id;
        PhysicalInstance inst;
        unsigned long long create, destroy;
      };
      struct MessageInfo {
      public:
        MessageKind kind;
        unsigned long long start, stop;
        Processor proc;
      };
      struct MapperCallInfo {
      public:
        MappingCallKind kind;
        UniqueID op_id;
        unsigned long long start, stop;
        Processor proc;
      };
      struct RuntimeCallInfo {
      public:
        RuntimeCallKind kind;
        unsigned long long start, stop;
        Processor proc;
      };
#ifdef LEGION_PROF_SELF_PROFILE
      struct ProfTaskInfo {
      public:
	Processor proc;
	UniqueID op_id;
	unsigned long long start, stop;
      };
#endif
    public:
      LegionProfInstance(LegionProfiler *owner);
      LegionProfInstance(const LegionProfInstance &rhs);
      ~LegionProfInstance(void);
    public:
      LegionProfInstance& operator=(const LegionProfInstance &rhs);
    public:
      void register_task_kind(TaskID task_id, const char *name, bool overwrite);
      void register_task_variant(TaskID task_id,
                                 VariantID variant_id, 
                                 const char *variant_name);
      void register_operation(Operation *op);
      void register_multi_task(Operation *op, TaskID kind);
      void register_slice_owner(UniqueID pid, UniqueID id);
    public:
      void process_task(VariantID variant_id, UniqueID op_id, 
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage,
                  Realm::ProfilingMeasurements::OperationEventWaits *waits);
      void process_meta(size_t id, UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage,
                  Realm::ProfilingMeasurements::OperationEventWaits *waits);
      void process_message(
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage,
                  Realm::ProfilingMeasurements::OperationEventWaits *waits);
      void process_copy(UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationMemoryUsage *usage);
      void process_fill(UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationMemoryUsage *usage);
      void process_inst_create(UniqueID op_id, PhysicalInstance inst,
		  unsigned long long create);
      void process_inst_usage(UniqueID op_id,
                  Realm::ProfilingMeasurements::InstanceMemoryUsage *usage);
      void process_inst_timeline(UniqueID op_id,
                  Realm::ProfilingMeasurements::InstanceTimeline *timeline);
    public:
      void record_message(Processor proc, MessageKind kind, 
                          unsigned long long start,
                          unsigned long long stop);
      void record_mapper_call(Processor proc, MappingCallKind kind, 
                          UniqueID uid, unsigned long long start,
                          unsigned long long stop);
      void record_runtime_call(Processor proc, RuntimeCallKind kind,
                          unsigned long long start, unsigned long long stop);
#ifdef LEGION_PROF_SELF_PROFILE
    public:
      void record_proftask(Processor p, UniqueID op_id,
			   unsigned long long start,
			   unsigned long long stop);
#endif
    public:
      void dump_state(void);
    private:
      LegionProfiler *const owner;
      std::deque<TaskKind>          task_kinds;
      std::deque<TaskVariant>       task_variants;
      std::deque<OperationInstance> operation_instances;
      std::deque<MultiTask>         multi_tasks;
      std::deque<SliceOwner>        slice_owners;
    private:
      std::deque<TaskInfo> task_infos;
      std::deque<MetaInfo> meta_infos;
      std::deque<CopyInfo> copy_infos;
      std::deque<FillInfo> fill_infos;
      std::deque<InstCreateInfo> inst_create_infos;
      std::deque<InstUsageInfo> inst_usage_infos;
      std::deque<InstTimelineInfo> inst_timeline_infos;
    private:
      std::deque<MessageInfo> message_infos;
      std::deque<MapperCallInfo> mapper_call_infos;
      std::deque<RuntimeCallInfo> runtime_call_infos;
#ifdef LEGION_PROF_SELF_PROFILE
    private:
      std::deque<ProfTaskInfo> prof_task_infos;
#endif
    };

    class LegionProfiler {
    public:
      enum ProfilingKind {
        LEGION_PROF_TASK,
        LEGION_PROF_META,
        LEGION_PROF_MESSAGE,
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
      void register_task_kind(TaskID task_id, const char *task_name, 
                              bool overwrite);
      void register_task_variant(TaskID task_id,
                                 VariantID var_id,
                                 const char *variant_name);
      // Operations
      void register_operation(Operation *op);
      void register_multi_task(Operation *op, TaskID task_id);
      void register_slice_owner(UniqueID pid, UniqueID id);
    public:
      void add_task_request(Realm::ProfilingRequestSet &requests, 
                            TaskID tid, SingleTask *task);
      void add_meta_request(Realm::ProfilingRequestSet &requests,
                            LgTaskID tid, Operation *op);
      void add_copy_request(Realm::ProfilingRequestSet &requests, 
                            Operation *op);
      void add_fill_request(Realm::ProfilingRequestSet &requests,
                            Operation *op);
      void add_inst_request(Realm::ProfilingRequestSet &requests,
                            Operation *op);
      // Adding a message profiling request is a static method
      // because we might not have a profiler on the local node
      static void add_message_request(Realm::ProfilingRequestSet &requests,
                            Processor remote_target);
    public:
      // Alternate versions of the one above with op ids
      void add_task_request(Realm::ProfilingRequestSet &requests, 
                            TaskID tid, UniqueID uid);
      void add_meta_request(Realm::ProfilingRequestSet &requests,
                            LgTaskID tid, UniqueID uid);
      void add_copy_request(Realm::ProfilingRequestSet &requests, 
                            UniqueID uid);
      void add_fill_request(Realm::ProfilingRequestSet &requests,
                            UniqueID uid);
      void add_inst_request(Realm::ProfilingRequestSet &requests,
                            UniqueID uid);
    public:
      // Process low-level runtime profiling results
      void process_results(Processor p, const void *buffer, size_t size);
    public:
      // Dump all the results
      void finalize(void);
    public:
      void record_instance_creation(PhysicalInstance inst, Memory memory,
                                    UniqueID op_id, unsigned long long create);
    public:
      void record_message_kinds(const char *const *const message_names,
                                unsigned int num_message_kinds);
      void record_message(MessageKind kind, unsigned long long start,
                          unsigned long long stop);
    public:
      void record_mapper_call_kinds(const char *const *const mapper_call_names,
                                    unsigned int num_mapper_call_kinds);
      void record_mapper_call(MappingCallKind kind, UniqueID uid,
                            unsigned long long start, unsigned long long stop);
    public:
      void record_runtime_call_kinds(const char *const *const runtime_calls,
                                     unsigned int num_runtime_call_kinds);
      void record_runtime_call(RuntimeCallKind kind,
                           unsigned long long start, unsigned long long stop);
    public:
      const Processor target_proc;
      inline bool has_outstanding_requests(void)
        { return total_outstanding_requests != 0; }
    public:
      inline void increment_total_outstanding_requests(void)
        { __sync_fetch_and_add(&total_outstanding_requests,1); }
      inline void decrement_total_outstanding_requests(void)
        { __sync_fetch_and_sub(&total_outstanding_requests,1); }
    private:
      void create_thread_local_profiling_instance(void);
    private:
      Reservation profiler_lock;
      std::vector<LegionProfInstance*> instances;
      unsigned total_outstanding_requests;
    };

    class DetailedProfiler {
    public:
      DetailedProfiler(Runtime *runtime, RuntimeCallKind call);
      DetailedProfiler(const DetailedProfiler &rhs);
      ~DetailedProfiler(void);
    public:
      DetailedProfiler& operator=(const DetailedProfiler &rhs);
    private:
      LegionProfiler *const profiler;
      const RuntimeCallKind call_kind;
      unsigned long long start_time;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_PROFILING_H__

