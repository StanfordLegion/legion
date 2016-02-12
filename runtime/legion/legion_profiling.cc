/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "legion_ops.h"
#include "legion_tasks.h"
#include "legion_profiling.h"

#include <cstring>
#include <cstdlib>

namespace Legion {
  namespace Internal {

    extern LegionRuntime::Logger::Category log_prof;

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
      log_prof.info("Prof User Info " IDFMT " %llu %llu %s", proc.id,
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
                                                const char *name)
    //--------------------------------------------------------------------------
    {
      task_kinds.push_back(TaskKind());
      TaskKind &kind = task_kinds.back();
      kind.task_id = task_id;
      kind.task_name = strdup(name);
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
      var.variant_name = strdup(variant_name);
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      operation_instances.push_back(OperationInstance());
      OperationInstance &inst = operation_instances.back();
      inst.op_id = op->get_unique_op_id();
      inst.op_kind = op->get_operation_kind();
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
    void LegionProfInstance::process_task(VariantID variant_id, UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(timeline->is_valid());
#endif
      task_infos.push_back(TaskInfo()); 
      TaskInfo &info = task_infos.back();
      info.op_id = op_id;
      info.variant_id = variant_id;
      info.proc = usage->proc;
      info.create = timeline->create_time;
      info.ready = timeline->ready_time;
      info.start = timeline->start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline->complete_time;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_meta(size_t id, UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(timeline->is_valid());
#endif
      meta_infos.push_back(MetaInfo());
      MetaInfo &info = meta_infos.back();
      info.op_id = op_id;
      info.hlr_id = id;
      info.proc = usage->proc;
      info.create = timeline->create_time;
      info.ready = timeline->ready_time;
      info.start = timeline->start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline->complete_time;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_copy(UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationMemoryUsage *usage)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(timeline->is_valid());
#endif
      copy_infos.push_back(CopyInfo());
      CopyInfo &info = copy_infos.back();
      info.op_id = op_id;
      info.source = usage->source;
      info.target = usage->target;
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
#ifdef DEBUG_HIGH_LEVEL
      assert(timeline->is_valid());
#endif
      fill_infos.push_back(FillInfo());
      FillInfo &info = fill_infos.back();
      info.op_id = op_id;
      info.target = usage->target;
      info.create = timeline->create_time;
      info.ready = timeline->ready_time;
      info.start = timeline->start_time;
      // use complete_time instead of end_time to include async work
      info.stop = timeline->complete_time;
    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_inst(UniqueID op_id,
                  Realm::ProfilingMeasurements::InstanceTimeline *timeline,
                  Realm::ProfilingMeasurements::InstanceMemoryUsage *usage)
    //--------------------------------------------------------------------------
    {
      inst_infos.push_back(InstInfo());
      InstInfo &info = inst_infos.back();
      info.op_id = op_id;
      info.inst = usage->instance;
      info.mem = usage->memory;
      info.total_bytes = usage->bytes;
      info.create = timeline->create_time;
      info.destroy = timeline->delete_time;
    }

#ifdef LEGION_PROF_MESSAGES
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
      info.proc = proc;
    }
#endif

    //--------------------------------------------------------------------------
    void LegionProfInstance::dump_state(void)
    //--------------------------------------------------------------------------
    {
      for (std::deque<TaskKind>::const_iterator it = task_kinds.begin();
            it != task_kinds.end(); it++)
      {
        log_prof.info("Prof Task Kind %u %s", it->task_id, it->task_name);
        free(const_cast<char*>(it->task_name));
      }
      for (std::deque<TaskVariant>::const_iterator it = task_variants.begin();
            it != task_variants.end(); it++)
      {
        log_prof.info("Prof Task Variant %u %lu %s", it->task_id,
            it->variant_id, it->variant_name);
        free(const_cast<char*>(it->variant_name));
      }
      for (std::deque<OperationInstance>::const_iterator it = 
            operation_instances.begin(); it != operation_instances.end(); it++)
      {
        log_prof.info("Prof Operation %llu %u", it->op_id, it->op_kind);
      }
      for (std::deque<MultiTask>::const_iterator it = 
            multi_tasks.begin(); it != multi_tasks.end(); it++)
      {
        log_prof.info("Prof Multi %llu %u", it->op_id, it->task_id);
      }
      for (std::deque<TaskInfo>::const_iterator it = task_infos.begin();
            it != task_infos.end(); it++)
      {
        log_prof.info("Prof Task Info %llu %lu " IDFMT " %llu %llu %llu %llu",
                      it->op_id, it->variant_id, it->proc.id, 
                      it->create, it->ready, it->start, it->stop);
      }
      for (std::deque<MetaInfo>::const_iterator it = meta_infos.begin();
            it != meta_infos.end(); it++)
      {
        log_prof.info("Prof Meta Info %llu %u " IDFMT " %llu %llu %llu %llu",
                      it->op_id, it->hlr_id, it->proc.id,
                      it->create, it->ready, it->start, it->stop);
      }
      for (std::deque<CopyInfo>::const_iterator it = copy_infos.begin();
            it != copy_infos.end(); it++)
      {
        log_prof.info("Prof Copy Info %llu " IDFMT " " IDFMT " %llu"
                      " %llu %llu %llu %llu", it->op_id, it->source.id,
                    it->target.id, it->size, it->create, it->ready, it->start,
                    it->stop);
      }
      for (std::deque<FillInfo>::const_iterator it = fill_infos.begin();
            it != fill_infos.end(); it++)
      {
        log_prof.info("Prof Fill Info %llu " IDFMT 
                      " %llu %llu %llu %llu", it->op_id, it->target.id, 
                            it->create, it->ready, it->start, it->stop);
      }
      for (std::deque<InstInfo>::const_iterator it = inst_infos.begin();
            it != inst_infos.end(); it++)
      {
        log_prof.info("Prof Inst Info %llu " IDFMT " " IDFMT " %lu %llu %llu",
                      it->op_id, it->inst.id, it->mem.id, it->total_bytes,
                      it->create, it->destroy);
      }
#ifdef LEGION_PROF_MESSAGES
      for (std::deque<MessageInfo>::const_iterator it = message_infos.begin();
            it != message_infos.end(); it++)
      {
        log_prof.info("Prof Message Info %u " IDFMT " %llu %llu",
                      it->kind, it->proc.id, it->start, it->stop);
      }
#endif
      task_kinds.clear();
      task_variants.clear();
      operation_instances.clear();
      multi_tasks.clear();
      task_infos.clear();
      meta_infos.clear();
      copy_infos.clear();
      inst_infos.clear();
#ifdef LEGION_PROF_MESSAGES
      message_infos.clear();
#endif
    }

    //--------------------------------------------------------------------------
    LegionProfiler::LegionProfiler(Processor target, const Machine &machine,
                                   unsigned num_meta_tasks,
                                   const char *const *const task_descriptions,
                                   unsigned num_operation_kinds,
                                   const char *const *const 
                                                  operation_kind_descriptions)
      : target_proc(target), instances((LegionProfInstance**)
            malloc(MAX_NUM_PROCS*sizeof(LegionProfInstance*))),
        total_outstanding_requests(0)
    //--------------------------------------------------------------------------
    {
      // Allocate space for all the instances and null it out
      for (unsigned idx = 0; idx < MAX_NUM_PROCS; idx++)
        instances[idx] = NULL;
      for (unsigned idx = 0; idx < num_meta_tasks; idx++)
      {
        log_prof.info("Prof Meta Desc %u %s", idx, task_descriptions[idx]);
      }
      for (unsigned idx = 0; idx < num_operation_kinds; idx++)
      {
        log_prof.info("Prof Op Desc %u %s", 
                        idx, operation_kind_descriptions[idx]);
      }
      // Log all the processors and memories
      std::set<Processor> all_procs;
      machine.get_all_processors(all_procs);
      for (std::set<Processor>::const_iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        log_prof.info("Prof Proc Desc " IDFMT " %d", it->id, it->kind());
      }
      std::set<Memory> all_mems;
      machine.get_all_memories(all_mems);
      for (std::set<Memory>::const_iterator it = all_mems.begin();
            it != all_mems.end(); it++)
      {
        log_prof.info("Prof Mem Desc " IDFMT " %d %ld", 
                      it->id, it->kind(), it->capacity());
      }
    }

    //--------------------------------------------------------------------------
    LegionProfiler::LegionProfiler(const LegionProfiler &rhs)
      : target_proc(rhs.target_proc), instances(rhs.instances)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LegionProfiler::~LegionProfiler(void)
    //--------------------------------------------------------------------------
    {
      assert(total_outstanding_requests == 0);
      for (unsigned idx = 0; idx < MAX_NUM_PROCS; idx++)
      {
        if (instances[idx] != NULL)
          delete instances[idx];
      }
      free(instances);
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
                                            const char *task_name)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      size_t local_id = current.local_id(); 
#ifdef DEBUG_HIGH_LEVEL
      assert(local_id < MAX_NUM_PROCS);
#endif
      if (instances[local_id] == NULL)
        instances[local_id] = new LegionProfInstance(this);
      instances[local_id]->register_task_kind(task_id, task_name);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_task_variant(TaskID task_id,
                                               VariantID variant_id, 
                                               const char *variant_name)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      size_t local_id = current.local_id(); 
#ifdef DEBUG_HIGH_LEVEL
      assert(local_id < MAX_NUM_PROCS);
#endif
      if (instances[local_id] == NULL)
        instances[local_id] = new LegionProfInstance(this);
      instances[local_id]->register_task_variant(task_id, variant_id,
          variant_name);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_operation(Operation *op)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      size_t local_id = current.local_id(); 
#ifdef DEBUG_HIGH_LEVEL
      assert(local_id < MAX_NUM_PROCS);
#endif
      if (instances[local_id] == NULL)
        instances[local_id] = new LegionProfInstance(this);
      instances[local_id]->register_operation(op);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::register_multi_task(Operation *op, TaskID task_id)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      size_t local_id = current.local_id(); 
#ifdef DEBUG_HIGH_LEVEL
      assert(local_id < MAX_NUM_PROCS);
#endif
      if (instances[local_id] == NULL)
        instances[local_id] = new LegionProfInstance(this);
      instances[local_id]->register_multi_task(op, task_id);
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
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_meta_request(Realm::ProfilingRequestSet &requests,
                                          HLRTaskID tid, Operation *op)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests();
      ProfilingInfo info(LEGION_PROF_META); 
      info.id = tid;
      info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>();
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
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
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
      ProfilingInfo info(LEGION_PROF_FILL);
      // No ID here
      info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
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
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
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
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_meta_request(Realm::ProfilingRequestSet &requests,
                                          HLRTaskID tid, UniqueID uid)
    //--------------------------------------------------------------------------
    {
      increment_total_outstanding_requests();
      ProfilingInfo info(LEGION_PROF_META); 
      info.id = tid;
      info.op_id = uid;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationProcessorUsage>();
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
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
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
      ProfilingInfo info(LEGION_PROF_FILL);
      // No ID here
      info.op_id = uid;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists())
                        ? target_proc : Processor::get_executing_processor(),
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
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
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists()) 
                        ? target_proc : Processor::get_executing_processor(),
                        HLR_LEGION_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::InstanceTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::InstanceMemoryUsage>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::process_results(Processor p, const void *buffer,
                                         size_t size)
    //--------------------------------------------------------------------------
    {
      size_t local_id = p.local_id(); 
#ifdef DEBUG_HIGH_LEVEL
      assert(local_id < MAX_NUM_PROCS);
#endif
      if (instances[local_id] == NULL)
        instances[local_id] = new LegionProfInstance(this);
      Realm::ProfilingResponse response(buffer, size);
#ifdef DEBUG_HIGH_LEVEL
      assert(response.user_data_size() == sizeof(ProfilingInfo));
#endif
      const ProfilingInfo *info = (const ProfilingInfo*)response.user_data();
      switch (info->kind)
      {
        case LEGION_PROF_TASK:
          {
#ifdef DEBUG_HIGH_LEVEL
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
            instances[local_id]->process_task(info->id, info->op_id,
                                              timeline, usage);
            delete timeline;
            delete usage;
            decrement_total_outstanding_requests();
            break;
          }
        case LEGION_PROF_META:
          {
#ifdef DEBUG_HIGH_LEVEL
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
            instances[local_id]->process_meta(info->id, info->op_id,
                                              timeline, usage);
            delete timeline;
            delete usage;
            decrement_total_outstanding_requests();
            break;
          }
        case LEGION_PROF_COPY:
          {
#ifdef DEBUG_HIGH_LEVEL
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
            instances[local_id]->process_copy(info->op_id,
                                              timeline, usage);
            delete timeline;
            delete usage;
            break;
          }
        case LEGION_PROF_FILL:
          {
#ifdef DEBUG_HIGH_LEVEL
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
            instances[local_id]->process_fill(info->op_id,
                                              timeline, usage);
            delete timeline;
            delete usage;
            break;
          }
        case LEGION_PROF_INST:
          {
#ifdef DEBUG_HIGH_LEVEL
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::InstanceTimeline>());
            assert(response.has_measurement<
                Realm::ProfilingMeasurements::InstanceMemoryUsage>());
#endif
            Realm::ProfilingMeasurements::InstanceTimeline *timeline = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::InstanceTimeline>();
            Realm::ProfilingMeasurements::InstanceMemoryUsage *usage = 
              response.get_measurement<
                    Realm::ProfilingMeasurements::InstanceMemoryUsage>();
            instances[local_id]->process_inst(info->op_id,
                                              timeline, usage);
            delete timeline;
            delete usage;
            break;
          }
        default:
          assert(false);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::finalize(void)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < MAX_NUM_PROCS; idx++)
      {
        if (instances[idx] != NULL)
          instances[idx]->dump_state();
      }
    }

#ifdef LEGION_PROF_MESSAGES
    //--------------------------------------------------------------------------
    void LegionProfiler::record_message_kinds(const char *const *const
                                  message_names, unsigned int num_message_kinds)
    //--------------------------------------------------------------------------
    {
      for (unsigned idx = 0; idx < num_message_kinds; idx++)
      {
        log_prof.info("Prof Message Desc %u %s", idx, message_names[idx]);
      }
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::record_message(MessageKind kind, 
                                        unsigned long long start,
                                        unsigned long long stop)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      size_t local_id = current.local_id();
#ifdef DEBUG_HIGH_LEVEL
      assert(local_id < MAX_NUM_PROCS);
#endif
      if (instances[local_id] == NULL)
        instances[local_id] = new LegionProfInstance(this);
      instances[local_id]->record_message(current, kind, start, stop);
    }
#endif

  }; // namespace Internal
}; // namespace Legion

