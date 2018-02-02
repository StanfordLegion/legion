/* Copyright 2018 Stanford University, NVIDIA Corporation
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
#include "legion/legion_types.h"
#include "legion/legion_utilities.h"
#include "realm/profiling.h"

#include <assert.h>
#include <deque>
#include <algorithm>
#include <sstream>

#define LEGION_PROF_SELF_PROFILE

#ifdef DETAILED_LEGION_PROF
#define DETAILED_PROFILER(runtime, call) \
  DetailedProfiler __detailed_profiler(runtime, call)
#else
#define DETAILED_PROFILER(runtime, call) // Nothing
#endif

namespace Legion {
  namespace Internal { 

    // XXX: Make sure these typedefs are consistent with Realm
    typedef ::realm_barrier_timestamp_t timestamp_t;
    typedef Realm::Processor::Kind ProcKind;
    typedef Realm::Memory::Kind MemKind;
    typedef ::realm_id_t ProcID;
    typedef ::realm_id_t MemID;
    typedef ::realm_id_t InstID;

    class LegionProfSerializer; // forward declaration

    class LegionProfMarker {
    public:
      LegionProfMarker(const char* name);
      ~LegionProfMarker();
      void mark_stop();
    private:
      const char* name;
      bool stopped;
      Processor proc;
      timestamp_t start, stop;
    };

    class LegionProfDesc {
    public:
      struct MessageDesc {
      public:
        unsigned kind;
        const char *name;
      };
      struct MapperCallDesc {
      public:
        unsigned kind;
        const char *name;
      };
      struct RuntimeCallDesc {
      public:
        unsigned kind;
        const char *name;
      };
      struct MetaDesc {
      public:
        unsigned kind;
        const char *name;
      };
      struct OpDesc {
      public:
        unsigned kind;
        const char *name;
      };
      struct ProcDesc {
      public:
        ProcID proc_id;
        ProcKind kind;
      };
      struct MemDesc {
      public:
        MemID mem_id;
        MemKind kind;
        unsigned long long capacity;
      };
    };

    class LegionProfInstance {
    public:
      struct TaskKind {
      public:
        TaskID task_id;
        const char *name;
        bool overwrite;
      };
      struct TaskVariant {
      public:
        TaskID task_id;
        VariantID variant_id;
        const char *name;
      };
      struct OperationInstance {
      public:
        UniqueID op_id;
        unsigned kind;
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
        timestamp_t wait_start, wait_ready, wait_end;
      };
      struct TaskInfo {
      public:
        UniqueID op_id;
        TaskID task_id;
        VariantID variant_id;
        ProcID proc_id;
        timestamp_t create, ready, start, stop;
        std::deque<WaitInfo> wait_intervals;
      };
      struct MetaInfo {
      public:
        UniqueID op_id;
        unsigned lg_id;
        ProcID proc_id;
        timestamp_t create, ready, start, stop;
        std::deque<WaitInfo> wait_intervals;
      };
      struct CopyInfo {
      public:
        UniqueID op_id;
        MemID src, dst;
        unsigned long long size;
        timestamp_t create, ready, start, stop;
      };
      struct FillInfo {
      public:
        UniqueID op_id;
        MemID dst;
        timestamp_t create, ready, start, stop;
      };
      struct InstCreateInfo {
      public:
        UniqueID op_id;
        InstID inst_id;
        timestamp_t create; // time of HLR creation request
      };
      struct InstUsageInfo {
      public:
        UniqueID op_id;
        InstID inst_id;
        MemID mem_id;
        unsigned long long size;
      };
      struct InstTimelineInfo {
      public:
        UniqueID op_id;
        InstID inst_id;
        timestamp_t create, destroy;
      };
      struct PartitionInfo {
      public:
        UniqueID op_id;
        DepPartOpKind part_op;
        unsigned long long create, ready, start, stop;
      };
      struct MessageInfo {
      public:
        MessageKind kind;
        timestamp_t start, stop;
        ProcID proc_id;
      };
      struct MapperCallInfo {
      public:
        MappingCallKind kind;
        UniqueID op_id;
        timestamp_t start, stop;
        ProcID proc_id;
      };
      struct RuntimeCallInfo {
      public:
        RuntimeCallKind kind;
        timestamp_t start, stop;
        ProcID proc_id;
      };
#ifdef LEGION_PROF_SELF_PROFILE
      struct ProfTaskInfo {
      public:
        ProcID proc_id;
        UniqueID op_id;
        timestamp_t start, stop;
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
      void process_task(TaskID task_id, VariantID variant_id, UniqueID op_id, 
            const Realm::ProfilingMeasurements::OperationTimeline &timeline,
            const Realm::ProfilingMeasurements::OperationProcessorUsage &usage,
            const Realm::ProfilingMeasurements::OperationEventWaits &waits);
      void process_meta(size_t id, UniqueID op_id,
            const Realm::ProfilingMeasurements::OperationTimeline &timeline,
            const Realm::ProfilingMeasurements::OperationProcessorUsage &usage,
            const Realm::ProfilingMeasurements::OperationEventWaits &waits);
      void process_message(
            const Realm::ProfilingMeasurements::OperationTimeline &timeline,
            const Realm::ProfilingMeasurements::OperationProcessorUsage &usage,
            const Realm::ProfilingMeasurements::OperationEventWaits &waits);
      void process_copy(UniqueID op_id,
            const Realm::ProfilingMeasurements::OperationTimeline &timeline,
            const Realm::ProfilingMeasurements::OperationMemoryUsage &usage);
      void process_fill(UniqueID op_id,
            const Realm::ProfilingMeasurements::OperationTimeline &timeline,
            const Realm::ProfilingMeasurements::OperationMemoryUsage &usage);
      void process_inst_create(UniqueID op_id, PhysicalInstance inst,
                               timestamp_t create);
      void process_inst_usage(UniqueID op_id,
            const Realm::ProfilingMeasurements::InstanceMemoryUsage &usage);
      void process_inst_timeline(UniqueID op_id,
            const Realm::ProfilingMeasurements::InstanceTimeline &timeline);
      void process_partition(UniqueID op_id, DepPartOpKind part_op,
            const Realm::ProfilingMeasurements::OperationTimeline &timeline);
    public:
      void record_message(Processor proc, MessageKind kind, timestamp_t start,
                          timestamp_t stop);
      void record_mapper_call(Processor proc, MappingCallKind kind, 
                              UniqueID uid, timestamp_t start,
                              timestamp_t stop);
      void record_runtime_call(Processor proc, RuntimeCallKind kind,
                               timestamp_t start, timestamp_t stop);
#ifdef LEGION_PROF_SELF_PROFILE
    public:
      void record_proftask(Processor p, UniqueID op_id, timestamp_t start,
                           timestamp_t stop);
#endif
    public:
      void dump_state(LegionProfSerializer *serializer);
      size_t dump_inter(LegionProfSerializer *serializer, const double over);
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
      std::deque<PartitionInfo> partition_infos;
    private:
      std::deque<MessageInfo> message_infos;
      std::deque<MapperCallInfo> mapper_call_infos;
      std::deque<RuntimeCallInfo> runtime_call_infos;
#ifdef LEGION_PROF_SELF_PROFILE
    private:
      std::deque<ProfTaskInfo> prof_task_infos;
#endif
    };

    class LegionProfiler : public ProfilingResponseHandler {
    public:
      enum ProfilingKind {
        LEGION_PROF_TASK,
        LEGION_PROF_META,
        LEGION_PROF_MESSAGE,
        LEGION_PROF_COPY,
        LEGION_PROF_FILL,
        LEGION_PROF_INST,
        LEGION_PROF_PARTITION,
        LEGION_PROF_LAST,
      };
      struct ProfilingInfo : public ProfilingResponseBase {
      public:
        ProfilingInfo(LegionProfiler *p, ProfilingKind k)
          : ProfilingResponseBase(p), kind(k) { }
      public:
        ProfilingKind kind;
        size_t id, id2;
        UniqueID op_id;
      };
    public:
      // Statically known information passed through the constructor
      // so that it can be deduplicated
      LegionProfiler(Processor target_proc, const Machine &machine,
                     Runtime *rt, unsigned num_meta_tasks,
                     const char *const *const meta_task_descriptions,
                     unsigned num_operation_kinds,
                     const char *const *const operation_kind_descriptions,
                     const char *serializer_type,
                     const char *prof_logname,
                     const size_t total_runtime_instances,
                     const size_t footprint_threshold,
                     const size_t target_latency);
      LegionProfiler(const LegionProfiler &rhs);
      virtual ~LegionProfiler(void);
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
                            TaskID tid, VariantID vid, SingleTask *task);
      void add_meta_request(Realm::ProfilingRequestSet &requests,
                            LgTaskID tid, Operation *op);
      void add_copy_request(Realm::ProfilingRequestSet &requests, 
                            Operation *op);
      void add_fill_request(Realm::ProfilingRequestSet &requests,
                            Operation *op);
      void add_inst_request(Realm::ProfilingRequestSet &requests,
                            Operation *op);
      void add_partition_request(Realm::ProfilingRequestSet &requests,
                                 Operation *op, DepPartOpKind part_op);
      // Adding a message profiling request is a static method
      // because we might not have a profiler on the local node
      static void add_message_request(Realm::ProfilingRequestSet &requests,
                                      Processor remote_target);
    public:
      // Alternate versions of the one above with op ids
      void add_task_request(Realm::ProfilingRequestSet &requests, 
                            TaskID tid, VariantID vid, UniqueID uid);
      void add_meta_request(Realm::ProfilingRequestSet &requests,
                            LgTaskID tid, UniqueID uid);
      void add_copy_request(Realm::ProfilingRequestSet &requests, 
                            UniqueID uid);
      void add_fill_request(Realm::ProfilingRequestSet &requests,
                            UniqueID uid);
      void add_inst_request(Realm::ProfilingRequestSet &requests,
                            UniqueID uid);
      void add_partition_request(Realm::ProfilingRequestSet &requests,
                                 UniqueID uid, DepPartOpKind part_op);
    public:
      // Process low-level runtime profiling results
      virtual void handle_profiling_response(
                            const Realm::ProfilingResponse &response);
    public:
      // Dump all the results
      void finalize(void);
    public:
      void record_instance_creation(PhysicalInstance inst, Memory memory,
                                    UniqueID op_id, timestamp_t create);
    public:
      void record_message_kinds(const char *const *const message_names,
                                unsigned int num_message_kinds);
      void record_message(MessageKind kind, timestamp_t start,
                          timestamp_t stop);
    public:
      void record_mapper_call_kinds(const char *const *const mapper_call_names,
                                    unsigned int num_mapper_call_kinds);
      void record_mapper_call(MappingCallKind kind, UniqueID uid,
                              timestamp_t start, timestamp_t stop);
    public:
      void record_runtime_call_kinds(const char *const *const runtime_calls,
                                     unsigned int num_runtime_call_kinds);
      void record_runtime_call(RuntimeCallKind kind, timestamp_t start,
                               timestamp_t stop);
    public:
#ifdef DEBUG_LEGION
      void increment_total_outstanding_requests(ProfilingKind kind,
                                                unsigned cnt = 1);
      void decrement_total_outstanding_requests(ProfilingKind kind,
                                                unsigned cnt = 1);
#else
      void increment_total_outstanding_requests(unsigned cnt = 1);
      void decrement_total_outstanding_requests(unsigned cnt = 1);
#endif
    public:
      void update_footprint(size_t diff, LegionProfInstance *inst);
    private:
      void create_thread_local_profiling_instance(void);
    public:
      Runtime *const runtime;
      // Event to trigger once the profiling is actually done
      const RtUserEvent done_event;
      // Size in bytes of the footprint before we start dumping
      const size_t output_footprint_threshold;
      // The goal size in microseconds of the output tasks
      const long long output_target_latency;
      // Target processor on which to launch jobs
      const Processor target_proc;
    private:
      LegionProfSerializer* serializer;
      mutable LocalLock profiler_lock;
      std::vector<LegionProfInstance*> instances;
#ifdef DEBUG_LEGION
      unsigned total_outstanding_requests[LEGION_PROF_LAST];
#else
      unsigned total_outstanding_requests;
#endif
    private:
      // For knowing when we need to start dumping early
      size_t total_memory_footprint;
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
      timestamp_t start_time;
    };

  }; // namespace Internal
}; // namespace Legion

#endif // __LEGION_PROFILING_H__

