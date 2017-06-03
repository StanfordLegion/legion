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

#include "legion.h"
#include "cmdline.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "legion_profiling.h"
#include "legion_profiling_serializer.h"

#include <cstring>
#include <cstdlib>

namespace Legion {
  namespace Internal {

    extern Realm::Logger log_prof;

    // Keep a thread-local profiler instance so we can always
    // be thread safe no matter what Realm decides to do 
    __thread LegionProfInstance *thread_local_profiling_instance = NULL;

    //--------------------------------------------------------------------------
    LegionProfMarker::LegionProfMarker(const char* _name)
      : name(_name), stopped(false)
    //--------------------------------------------------------------------------
    {
      proc = Realm::Processor::get_executing_processor();
      start = Realm::Clock::current_time_in_nanoseconds();
    }

    //--------------------------------------------------------------------------
    LegionProfMarker::~LegionProfMarker()
    //--------------------------------------------------------------------------
    {
      if (!stopped) mark_stop();
      log_prof.print("Prof User Info " IDFMT " %llu %llu %s", proc.id,
		     start, stop, name);
    }

    //--------------------------------------------------------------------------
    void LegionProfMarker::mark_stop()
    //--------------------------------------------------------------------------
    {
      stop = Realm::Clock::current_time_in_nanoseconds();
      stopped = true;
    }

    //--------------------------------------------------------------------------
    LegionProfInstance::LegionProfInstance(LegionProfiler *own)
      : owner(own)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LegionProfInstance::LegionProfInstance(const LegionProfInstance &rhs)
      : owner(rhs.owner)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LegionProfInstance::~LegionProfInstance(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LegionProfInstance& LegionProfInstance::operator=(
                                                  const LegionProfInstance &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_task_kind(TaskID task_id,
                                                const char *name,bool overwrite)
    //--------------------------------------------------------------------------
    {
      task_kinds.push_back(TaskKind());
      TaskKind &kind = task_kinds.back();
      kind.task_id = task_id;
      kind.name = strdup(name);
      kind.overwrite = overwrite;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_task_variant(TaskID task_id,
                                                   VariantID variant_id,
                                                   const char *variant_name)
    //--------------------------------------------------------------------------
    {
      task_variants.push_back(TaskVariant()); 
      TaskVariant &var = task_variants.back();
      var.task_id = task_id;
      var.variant_id = variant_id;
      var.name = strdup(variant_name);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      operation_instances.push_back(OperationInstance());
      OperationInstance &inst = operation_instances.back();
      inst.op_id = op->get_unique_op_id();
      inst.kind = op->get_operation_kind();
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_multi_task(Operation *op, TaskID task_id)
    //--------------------------------------------------------------------------
    {
      multi_tasks.push_back(MultiTask());
      MultiTask &task = multi_tasks.back();
      task.op_id = op->get_unique_op_id();
      task.task_id = task_id;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_slice_owner(UniqueID pid, UniqueID id)
    //--------------------------------------------------------------------------
    {
      slice_owners.push_back(SliceOwner());
      SliceOwner &task = slice_owners.back();
      task.parent_id = pid;
      task.op_id = id;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_task(VariantID variant_id, UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage,
                  Realm::ProfilingMeasurements::OperationEventWaits *waits)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(timeline->is_valid());
#endif
      task_infos.push_back(TaskInfo()); 
      TaskInfo &info = task_infos.back();
      info.op_id = op_id;
      info.variant_id = variant_id;
      info.proc_id = usage->proc.id;
      info.create = timeline->create_time;
      info.ready = timeline->ready_time;
      info.start = timeline->start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline->complete_time;
      unsigned num_intervals = waits->intervals.size();
      if (num_intervals > 0)
      {
        for (unsigned idx = 0; idx < num_intervals; ++idx)
        {
          info.wait_intervals.push_back(WaitInfo());
          WaitInfo& wait_info = info.wait_intervals.back();
          wait_info.wait_start = waits->intervals[idx].wait_start;
          wait_info.wait_ready = waits->intervals[idx].wait_ready;
          wait_info.wait_end = waits->intervals[idx].wait_end;
        }
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_meta(size_t id, UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage,
                  Realm::ProfilingMeasurements::OperationEventWaits *waits)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(timeline->is_valid());
#endif
      meta_infos.push_back(MetaInfo());
      MetaInfo &info = meta_infos.back();
      info.op_id = op_id;
      info.lg_id = id;
      info.proc_id = usage->proc.id;
      info.create = timeline->create_time;
      info.ready = timeline->ready_time;
      info.start = timeline->start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline->complete_time;
      unsigned num_intervals = waits->intervals.size();
      if (num_intervals > 0)
      {
        for (unsigned idx = 0; idx < num_intervals; ++idx)
        {
          info.wait_intervals.push_back(WaitInfo());
          WaitInfo& wait_info = info.wait_intervals.back();
          wait_info.wait_start = waits->intervals[idx].wait_start;
          wait_info.wait_ready = waits->intervals[idx].wait_ready;
          wait_info.wait_end = waits->intervals[idx].wait_end;
        }
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_message(
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage,
                  Realm::ProfilingMeasurements::OperationEventWaits *waits)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(timeline->is_valid());
#endif
      meta_infos.push_back(MetaInfo());
      MetaInfo &info = meta_infos.back();
      info.op_id = 0;
      info.lg_id = LG_MESSAGE_ID;
      info.proc_id = usage->proc.id;
      info.create = timeline->create_time;
      info.ready = timeline->ready_time;
      info.start = timeline->start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline->complete_time;
      unsigned num_intervals = waits->intervals.size();
      if (num_intervals > 0)
      {
        for (unsigned idx = 0; idx < num_intervals; ++idx)
        {
          info.wait_intervals.push_back(WaitInfo());
          WaitInfo& wait_info = info.wait_intervals.back();
          wait_info.wait_start = waits->intervals[idx].wait_start;
          wait_info.wait_ready = waits->intervals[idx].wait_ready;
          wait_info.wait_end = waits->intervals[idx].wait_end;
        }
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_copy(UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationMemoryUsage *usage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(timeline->is_valid());
#endif
      copy_infos.push_back(CopyInfo());
      CopyInfo &info = copy_infos.back();
      info.op_id = op_id;
      info.src = usage->source.id;
      info.dst = usage->target.id;
      info.size = usage->size;
      info.create = timeline->create_time;
      info.ready = timeline->ready_time;
      info.start = timeline->start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline->complete_time;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_fill(UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationMemoryUsage *usage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(timeline->is_valid());
#endif
      fill_infos.push_back(FillInfo());
      FillInfo &info = fill_infos.back();
      info.op_id = op_id;
      info.dst = usage->target.id;
      info.create = timeline->create_time;
      info.ready = timeline->ready_time;
      info.start = timeline->start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline->complete_time;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_inst_create(UniqueID op_id,
		  PhysicalInstance inst, unsigned long long create)
    //--------------------------------------------------------------------------
    {
      inst_create_infos.push_back(InstCreateInfo());
      InstCreateInfo &info = inst_create_infos.back();
      info.op_id = op_id;
      info.inst_id = inst.id;
      info.create = create;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_inst_usage(UniqueID op_id,
                  Realm::ProfilingMeasurements::InstanceMemoryUsage *usage)
    //--------------------------------------------------------------------------
    {
      inst_usage_infos.push_back(InstUsageInfo());
      InstUsageInfo &info = inst_usage_infos.back();
      info.op_id = op_id;
      info.inst_id = usage->instance.id;
      info.mem_id = usage->memory.id;
      info.size = usage->bytes;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_inst_timeline(UniqueID op_id,
                  Realm::ProfilingMeasurements::InstanceTimeline *timeline)
    //--------------------------------------------------------------------------
    {
      inst_timeline_infos.push_back(InstTimelineInfo());
      InstTimelineInfo &info = inst_timeline_infos.back();
      info.op_id = op_id;
      info.inst_id = timeline->instance.id;
      info.create = timeline->create_time;
      info.destroy = timeline->delete_time;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_message(Processor proc, MessageKind kind, 
                                            unsigned long long start,
                                            unsigned long long stop)
    //--------------------------------------------------------------------------
    {
      message_infos.push_back(MessageInfo());
      MessageInfo &info = message_infos.back();
      info.kind = kind;
      info.start = start;
      info.stop = stop;
      info.proc_id = proc.id;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_mapper_call(Processor proc, 
                              MappingCallKind kind, UniqueID uid,
                              unsigned long long start, unsigned long long stop)
    //--------------------------------------------------------------------------
    {
      mapper_call_infos.push_back(MapperCallInfo());
      MapperCallInfo &info = mapper_call_infos.back();
      info.kind = kind;
      info.op_id = uid;
      info.start = start;
      info.stop = stop;
      info.proc_id = proc.id;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::record_runtime_call(Processor proc, 
        RuntimeCallKind kind, unsigned long long start, unsigned long long stop)
    //--------------------------------------------------------------------------
    {
      runtime_call_infos.push_back(RuntimeCallInfo());
      RuntimeCallInfo &info = runtime_call_infos.back();
      info.kind = kind;
      info.start = start;
      info.stop = stop;
      info.proc_id = proc.id;
    }

#ifdef LEGION_PROF_SELF_PROFILE
    //--------------------------------------------------------------------------
    void LegionProfInstance::record_proftask(Processor proc, UniqueID op_id,
					     unsigned long long start,
					     unsigned long long stop)
    //--------------------------------------------------------------------------
    {
      prof_task_infos.push_back(ProfTaskInfo());
      ProfTaskInfo &info = prof_task_infos.back();
      info.proc_id = proc.id;
      info.op_id = op_id;
      info.start = start;
      info.stop = stop;
    }
#endif

    //--------------------------------------------------------------------------
    void LegionProfInstance::dump_state(LegionProfSerializer *serializer)
    //--------------------------------------------------------------------------
    {
      for (std::deque<TaskKind>::const_iterator it = task_kinds.begin();
            it != task_kinds.end(); it++)
      {
        serializer->serialize(*it);
        free(const_cast<char*>(it->name));
      }
      for (std::deque<TaskVariant>::const_iterator it = task_variants.begin();
            it != task_variants.end(); it++)
      {
        serializer->serialize(*it);
        free(const_cast<char*>(it->name));
      }
      for (std::deque<OperationInstance>::const_iterator it = 
            operation_instances.begin(); it != operation_instances.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<MultiTask>::const_iterator it = 
            multi_tasks.begin(); it != multi_tasks.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<SliceOwner>::const_iterator it = 
            slice_owners.begin(); it != slice_owners.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<TaskInfo>::const_iterator it = task_infos.begin();
            it != task_infos.end(); it++)
      {
        serializer->serialize(*it);
        for (std::deque<WaitInfo>::const_iterator wit =
             it->wait_intervals.begin(); wit != it->wait_intervals.end(); wit++)
        {
          serializer->serialize(*wit, *it);
        }
      }
      for (std::deque<MetaInfo>::const_iterator it = meta_infos.begin();
            it != meta_infos.end(); it++)
      {
        serializer->serialize(*it);
        for (std::deque<WaitInfo>::const_iterator wit =
             it->wait_intervals.begin(); wit != it->wait_intervals.end(); wit++)
        {
          serializer->serialize(*wit, *it);
        }
      }
      for (std::deque<CopyInfo>::const_iterator it = copy_infos.begin();
            it != copy_infos.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<FillInfo>::const_iterator it = fill_infos.begin();
            it != fill_infos.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<InstCreateInfo>::const_iterator it = 
            inst_create_infos.begin(); it != inst_create_infos.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<InstUsageInfo>::const_iterator it = 
            inst_usage_infos.begin(); it != inst_usage_infos.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<InstTimelineInfo>::const_iterator it = 
            inst_timeline_infos.begin(); it != inst_timeline_infos.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<MessageInfo>::const_iterator it = message_infos.begin();
            it != message_infos.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<MapperCallInfo>::const_iterator it = 
            mapper_call_infos.begin(); it != mapper_call_infos.end(); it++)
      {
        serializer->serialize(*it);
      }
      for (std::deque<RuntimeCallInfo>::const_iterator it = 
            runtime_call_infos.begin(); it != runtime_call_infos.end(); it++)
      {
        serializer->serialize(*it);
      }
#ifdef LEGION_PROF_SELF_PROFILE
      for (std::deque<ProfTaskInfo>::const_iterator it = 
            prof_task_infos.begin(); it != prof_task_infos.end(); it++)
      {
        serializer->serialize(*it);
      }
#endif
      task_kinds.clear();
      task_variants.clear();
      operation_instances.clear();
      multi_tasks.clear();
      task_infos.clear();
      meta_infos.clear();
      copy_infos.clear();
      inst_create_infos.clear();
      inst_usage_infos.clear();
      inst_timeline_infos.clear();
      message_infos.clear();
      mapper_call_infos.clear();
    }

    //--------------------------------------------------------------------------
    LegionProfiler::LegionProfiler(Processor target, const Machine &machine,
                                   unsigned num_meta_tasks,
                                   const char *const *const task_descriptions,
                                   unsigned num_operation_kinds,
                                   const char *const *const 
                                                  operation_kind_descriptions,
                                   const char *serializer_type,
                                   const char *prof_logfile)
      : target_proc(target), total_outstanding_requests(0)
    //--------------------------------------------------------------------------
    {
      profiler_lock = Reservation::create_reservation();

      if (!strcmp(serializer_type, "binary")) {
        if (prof_logfile == NULL) {
          fprintf(stderr, "ERROR: Please specify -lg:prof_logfile <logfile_name> "
                          "when running with -lg:serializer binary");
          exit(-1);
        }
        Processor current = Processor::get_executing_processor();
        std::stringstream ss;
        ss << prof_logfile << current.address_space();
        serializer = new LegionProfBinarySerializer(ss.str());
      } else if (!strcmp(serializer_type, "ascii")) {
        if (prof_logfile != NULL) {
          fprintf(stderr, "ERROR: You should not specify -lg:prof_logfile "
                          "<logfile_name> when running with -lg:serializer ascii\n"
                          "       legion_prof output will be written to -logfile "
                                  "<logfile_name> instead");
          exit(-1);
        }
        serializer = new LegionProfASCIISerializer();
      } else {
        fprintf(stderr, "Invalid serializer (%s), must be 'binary' or 'ascii'\n", 
                        serializer_type);
        exit(-1);
      }

      for (unsigned idx = 0; idx < num_meta_tasks; idx++)
      {
        LegionProfDesc::MetaDesc meta_desc;
        meta_desc.kind = idx;
        meta_desc.name = task_descriptions[idx];
        serializer->serialize(meta_desc);
      }
      for (unsigned idx = 0; idx < num_operation_kinds; idx++)
      {
        LegionProfDesc::OpDesc op_desc;
        op_desc.kind = idx;
        op_desc.name = operation_kind_descriptions[idx];
        serializer->serialize(op_desc);
      }
      // Log all the processors and memories
      Machine::ProcessorQuery all_procs(machine);
      for (Machine::ProcessorQuery::iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        LegionProfDesc::ProcDesc proc_desc;
        proc_desc.proc_id = it->id;
        proc_desc.kind = it->kind();
        serializer->serialize(proc_desc);
      }
      Machine::MemoryQuery all_mems(machine);
      for (Machine::MemoryQuery::iterator it = all_mems.begin();
            it != all_mems.end(); it++)
      {
        LegionProfDesc::MemDesc mem_desc;
        mem_desc.mem_id = it->id;
        mem_desc.kind = it->kind();
        mem_desc.capacity = it->capacity();
        serializer->serialize(mem_desc);
      }
    }

    //--------------------------------------------------------------------------
    LegionProfiler::LegionProfiler(const LegionProfiler &rhs)
      : target_proc(rhs.target_proc)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LegionProfiler::~LegionProfiler(void)
    //--------------------------------------------------------------------------
    {
      profiler_lock.destroy_reservation();
      profiler_lock = Reservation::NO_RESERVATION;
      assert(total_outstanding_requests == 0);
      for (std::vector<LegionProfInstance*>::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
        delete (*it);

      // remove our serializer
      delete serializer;
    }

    //--------------------------------------------------------------------------
    LegionProfiler& LegionProfiler::operator=(const LegionProfiler &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_task_kind(TaskID task_id,
                                          const char *task_name, bool overwrite)
    //--------------------------------------------------------------------------
    {
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      thread_local_profiling_instance->register_task_kind(task_id, task_name,
                                                          overwrite);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_task_variant(TaskID task_id,
                                               VariantID variant_id, 
                                               const char *variant_name)
    //--------------------------------------------------------------------------
    {
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      thread_local_profiling_instance->register_task_variant(task_id, 
                                                      variant_id, variant_name);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      thread_local_profiling_instance->register_operation(op);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_multi_task(Operation *op, TaskID task_id)
    //--------------------------------------------------------------------------
    {
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      thread_local_profiling_instance->register_multi_task(op, task_id);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_slice_owner(UniqueID pid, UniqueID id)
    //--------------------------------------------------------------------------
    {
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      thread_local_profiling_instance->register_slice_owner(pid, id);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_task_request(Realm::ProfilingRequestSet &requests,
                                          TaskID tid, SingleTask *task)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests();
      ProfilingInfo info(LEGION_PROF_TASK); 
      info.id = tid;
      info.op_id = task->get_unique_id();
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationEventWaits>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_meta_request(Realm::ProfilingRequestSet &requests,
                                          LgTaskID tid, Operation *op)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests();
      ProfilingInfo info(LEGION_PROF_META); 
      info.id = tid;
      info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationEventWaits>();
    }

    //--------------------------------------------------------------------------
    /*static*/ void LegionProfiler::add_message_request(
                  Realm::ProfilingRequestSet &requests, Processor remote_target)
    //--------------------------------------------------------------------------
    {
      // Don't increment here, we'll increment on the remote side since we
      // that is where we know the profiler is going to handle the results
      ProfilingInfo info(LEGION_PROF_MESSAGE);
      Realm::ProfilingRequest &req = requests.add_request(remote_target,
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationEventWaits>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_copy_request(Realm::ProfilingRequestSet &requests,
                                          Operation *op)
    //--------------------------------------------------------------------------
    {
      ProfilingInfo info(LEGION_PROF_COPY); 
      // No ID here
      info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationMemoryUsage>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_fill_request(Realm::ProfilingRequestSet &requests,
                                          Operation *op)
    //--------------------------------------------------------------------------
    {
      // wonchan: don't track fill operations for the moment
      // as their requests and responses do not exactly match
      //increment_total_outstanding_requests();
      ProfilingInfo info(LEGION_PROF_FILL);
      // No ID here
      info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationMemoryUsage>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_inst_request(Realm::ProfilingRequestSet &requests,
                                          Operation *op)
    //--------------------------------------------------------------------------
    {
      ProfilingInfo info(LEGION_PROF_INST); 
      // No ID here
      info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::InstanceTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::InstanceMemoryUsage>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_task_request(Realm::ProfilingRequestSet &requests,
                                          TaskID tid, UniqueID uid)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests();
      ProfilingInfo info(LEGION_PROF_TASK); 
      info.id = tid;
      info.op_id = uid;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationEventWaits>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_meta_request(Realm::ProfilingRequestSet &requests,
                                          LgTaskID tid, UniqueID uid)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests();
      ProfilingInfo info(LEGION_PROF_META); 
      info.id = tid;
      info.op_id = uid;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationEventWaits>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_copy_request(Realm::ProfilingRequestSet &requests,
                                          UniqueID uid)
    //--------------------------------------------------------------------------
    {
      ProfilingInfo info(LEGION_PROF_COPY); 
      // No ID here
      info.op_id = uid;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationMemoryUsage>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_fill_request(Realm::ProfilingRequestSet &requests,
                                          UniqueID uid)
    //--------------------------------------------------------------------------
    {
      // wonchan: don't track fill operations for the moment
      // as their requests and responses do not exactly match
      //increment_total_outstanding_requests();
      ProfilingInfo info(LEGION_PROF_FILL);
      // No ID here
      info.op_id = uid;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationMemoryUsage>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_inst_request(Realm::ProfilingRequestSet &requests,
                                          UniqueID uid)
    //--------------------------------------------------------------------------
    {
      ProfilingInfo info(LEGION_PROF_INST); 
      // No ID here
      info.op_id = uid;
      // Instances use two profiling requests so that we can get MemoryUsage
      // right away - the Timeline doesn't come until we delete the instance
      Processor p = (target_proc.exists() 
                        ? target_proc : Processor::get_executing_processor());
      Realm::ProfilingRequest &req1 = requests.add_request(p,
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req1.add_measurement<
                 Realm::ProfilingMeasurements::InstanceMemoryUsage>();
      Realm::ProfilingRequest &req2 = requests.add_request(p,
                        LG_LEGION_PROFILING_ID, &info, sizeof(info));
      req2.add_measurement<
                 Realm::ProfilingMeasurements::InstanceTimeline>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::process_results(Processor p, const void *buffer,
                                         size_t size)
    //--------------------------------------------------------------------------
    {
#ifdef LEGION_PROF_SELF_PROFILE
      long long t_start = Realm::Clock::current_time_in_nanoseconds();
#endif
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      Realm::ProfilingResponse response(buffer, size);
#ifdef DEBUG_LEGION
      assert(response.user_data_size() == sizeof(ProfilingInfo));
#endif
      const ProfilingInfo *info = (const ProfilingInfo*)response.user_data();
      switch (info->kind)
      {
        case LEGION_PROF_TASK:
          {
#ifdef DEBUG_LEGION
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>());
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>());
#endif
            Realm::ProfilingMeasurements::OperationTimeline *timeline = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>();
            Realm::ProfilingMeasurements::OperationProcessorUsage *usage = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationProcessorUsage>();
            Realm::ProfilingMeasurements::OperationEventWaits *waits = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationEventWaits>();
            // Ignore anything that was predicated false for now
            if (usage != NULL)
              thread_local_profiling_instance->process_task(info->id, 
                  info->op_id, timeline, usage, waits);
            if (timeline != NULL)
              delete timeline;
            if (timeline != NULL)
              delete usage;
            decrement_total_outstanding_requests();
            break;
          }
        case LEGION_PROF_META:
          {
#ifdef DEBUG_LEGION
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>());
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>());
#endif
            Realm::ProfilingMeasurements::OperationTimeline *timeline = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>();
            Realm::ProfilingMeasurements::OperationProcessorUsage *usage = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationProcessorUsage>();
            Realm::ProfilingMeasurements::OperationEventWaits *waits = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationEventWaits>();
            // Ignore anything that was predicated false for now
            if (usage != NULL)
              thread_local_profiling_instance->process_meta(info->id, 
                  info->op_id, timeline, usage, waits);
            if (timeline != NULL)
              delete timeline;
            if (usage != NULL)
              delete usage;
            decrement_total_outstanding_requests();
            break;
          }
        case LEGION_PROF_MESSAGE:
          {
#ifdef DEBUG_LEGION
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>());
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>());
#endif
            Realm::ProfilingMeasurements::OperationTimeline *timeline = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>();
            Realm::ProfilingMeasurements::OperationProcessorUsage *usage = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationProcessorUsage>();
            Realm::ProfilingMeasurements::OperationEventWaits *waits = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationEventWaits>();
            if (usage != NULL)
              thread_local_profiling_instance->process_message(timeline, 
                  usage, waits);
            if (timeline != NULL)
              delete timeline;
            if (usage != NULL)
              delete usage;
            decrement_total_outstanding_requests();
            break;
          }
        case LEGION_PROF_COPY:
          {
#ifdef DEBUG_LEGION
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>());
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationMemoryUsage>());
#endif
            Realm::ProfilingMeasurements::OperationTimeline *timeline = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>();
            Realm::ProfilingMeasurements::OperationMemoryUsage *usage = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationMemoryUsage>();
            // Ignore anything that was predicated false for now
            if (usage != NULL)
              thread_local_profiling_instance->process_copy(info->op_id,
                                                            timeline, usage);
            if (timeline != NULL)
              delete timeline;
            if (usage != NULL)
              delete usage;
            break;
          }
        case LEGION_PROF_FILL:
          {
#ifdef DEBUG_LEGION
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>());
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::OperationMemoryUsage>());
#endif
            Realm::ProfilingMeasurements::OperationTimeline *timeline = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationTimeline>();
            Realm::ProfilingMeasurements::OperationMemoryUsage *usage = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::OperationMemoryUsage>();
            // Ignore anything that was predicated false for now
            if (usage != NULL)
              thread_local_profiling_instance->process_fill(info->op_id,
                                                            timeline, usage);
            if (timeline != NULL)
              delete timeline;
            if (usage != NULL)
              delete usage;
            // wonchan: don't track fill operations for the moment
            // as their requests and responses do not exactly match
            //decrement_total_outstanding_requests();
            break;
          }
        case LEGION_PROF_INST:
          {
	    // Record data based on which measurements we got back this time
	    if (response.has_measurement<
                Realm::ProfilingMeasurements::InstanceTimeline>())
	    {
	      Realm::ProfilingMeasurements::InstanceTimeline *timeline = 
                response.get_measurement<
                      Realm::ProfilingMeasurements::InstanceTimeline>();
	      thread_local_profiling_instance->process_inst_timeline(
								info->op_id,
								timeline);
	      delete timeline;
	    }
	    if (response.has_measurement<
                Realm::ProfilingMeasurements::InstanceMemoryUsage>())
	    {
	      Realm::ProfilingMeasurements::InstanceMemoryUsage *usage = 
                response.get_measurement<
                      Realm::ProfilingMeasurements::InstanceMemoryUsage>();
	      thread_local_profiling_instance->process_inst_usage(info->op_id,
								  usage);
	      delete usage;
	    }
            break;
          }
        default:
          assert(false);
      }
#ifdef LEGION_PROF_SELF_PROFILE
      long long t_stop = Realm::Clock::current_time_in_nanoseconds();
      thread_local_profiling_instance->record_proftask(p, info->op_id, 
                                                       t_start, t_stop);
#endif
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::finalize(void)
    //--------------------------------------------------------------------------
    {
      for (std::vector<LegionProfInstance*>::const_iterator it = 
            instances.begin(); it != instances.end(); it++) {
        (*it)->dump_state(serializer);
      }  
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_instance_creation(PhysicalInstance inst,
                       Memory memory, UniqueID op_id, unsigned long long create)
    //--------------------------------------------------------------------------
    {
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      thread_local_profiling_instance->process_inst_create(op_id, inst, create);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_message_kinds(const char *const *const
                                  message_names, unsigned int num_message_kinds)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < num_message_kinds; idx++)
      {
        LegionProfDesc::MessageDesc message_desc;
        message_desc.kind = idx;
        message_desc.name = message_names[idx];
        serializer->serialize(message_desc);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_message(MessageKind kind, 
                                        unsigned long long start,
                                        unsigned long long stop)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      thread_local_profiling_instance->record_message(current, kind, 
                                                      start, stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_mapper_call_kinds(const char *const *const
                               mapper_call_names, unsigned int num_mapper_calls)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < num_mapper_calls; idx++)
      {
        LegionProfDesc::MapperCallDesc mapper_call_desc;
        mapper_call_desc.kind = idx;
        mapper_call_desc.name = mapper_call_names[idx];
        serializer->serialize(mapper_call_desc);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_mapper_call(MappingCallKind kind, UniqueID uid,
                              unsigned long long start, unsigned long long stop)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      thread_local_profiling_instance->record_mapper_call(current, kind, uid, 
                                                   start, stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_runtime_call_kinds(const char *const *const
                             runtime_call_names, unsigned int num_runtime_calls)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < num_runtime_calls; idx++)
      {
        LegionProfDesc::RuntimeCallDesc runtime_call_desc;
        runtime_call_desc.kind = idx;
        runtime_call_desc.name = runtime_call_names[idx];
        serializer->serialize(runtime_call_desc);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_runtime_call(RuntimeCallKind kind,
                              unsigned long long start, unsigned long long stop)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      if (thread_local_profiling_instance == NULL)
        create_thread_local_profiling_instance();
      thread_local_profiling_instance->record_runtime_call(current, kind, 
                                                           start, stop);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::create_thread_local_profiling_instance(void)
    //--------------------------------------------------------------------------
    {
      thread_local_profiling_instance = new LegionProfInstance(this);
      // Task the lock and save the reference
      AutoLock p_lock(profiler_lock);
      instances.push_back(thread_local_profiling_instance);
    }

    //--------------------------------------------------------------------------
    DetailedProfiler::DetailedProfiler(Runtime *runtime, RuntimeCallKind call)
      : profiler(runtime->profiler), call_kind(call), start_time(0)
    //--------------------------------------------------------------------------
    {
      if (profiler != NULL)
        start_time = Realm::Clock::current_time_in_nanoseconds();
    }

    //--------------------------------------------------------------------------
    DetailedProfiler::DetailedProfiler(const DetailedProfiler &rhs)
      : profiler(rhs.profiler), call_kind(rhs.call_kind)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    DetailedProfiler::~DetailedProfiler(void)
    //--------------------------------------------------------------------------
    {
      if (profiler != NULL)
      {
        unsigned long long stop_time = 
          Realm::Clock::current_time_in_nanoseconds();
        profiler->record_runtime_call(call_kind, start_time, stop_time);
      }
    }

    //--------------------------------------------------------------------------
    DetailedProfiler& DetailedProfiler::operator=(const DetailedProfiler &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

  }; // namespace Internal
}; // namespace Legion

