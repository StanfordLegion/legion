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

#include "legion_ops.h"
#include "legion_tasks.h"
#include "legion_profiling.h"

namespace LegionRuntime {
  namespace HighLevel {

    //--------------------------------------------------------------------------
    LegionProfInstance::LegionProfInstance(LegionProfiler *own)
      : owner(own)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    LegionProfInstance::LegionProfInstance(const LegionProfInstance &rhs)
      : owner(NULL)
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
    void LegionProfInstance::register_task_variant(
                                        TaskVariantCollection::Variant *variant)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_operation(Operation *op)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::register_task(SingleTask *task)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_task(size_t id, UniqueID op_id, 
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_meta(size_t id, UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationProcessorUsage *usage)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_copy(UniqueID op_id,
                  Realm::ProfilingMeasurements::OperationTimeline *timeline,
                  Realm::ProfilingMeasurements::OperationMemoryUsage *usage)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    void LegionProfInstance::process_inst(size_t id, UniqueID op_id,
                  Realm::ProfilingMeasurements::InstanceTimeline *timeline,
                  Realm::ProfilingMeasurements::InstanceMemoryUsage *usage)
    //--------------------------------------------------------------------------
    {

    }

    //--------------------------------------------------------------------------
    LegionProfiler::LegionProfiler(Processor target,
                                   unsigned num_meta,
                                   const char *meta_descs,
                                   unsigned num_kinds,
                                   const char *op_descs)
      : target_proc(target), num_meta_tasks(num_meta), 
        task_descriptions(meta_descs), num_operation_kinds(num_kinds), 
        operation_kind_descriptions(op_descs), instances((LegionProfInstance**)
            malloc(MAX_NUM_PROCS*sizeof(LegionProfInstance*)))
    //--------------------------------------------------------------------------
    {
      // Allocate space for all the instances and null it out
      for (unsigned idx = 0; idx < MAX_NUM_PROCS; idx++)
        instances[idx] = NULL;
    }

    //--------------------------------------------------------------------------
    LegionProfiler::LegionProfiler(const LegionProfiler &rhs)
      : target_proc(rhs.target_proc), num_meta_tasks(0),
        task_descriptions(NULL), num_operation_kinds(0),
        operation_kind_descriptions(NULL), instances(NULL)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    LegionProfiler::~LegionProfiler(void)
    //--------------------------------------------------------------------------
    {
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
    void LegionProfiler::register_task_variant(
                                        TaskVariantCollection::Variant *variant)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      size_t local_id = current.local_id(); 
#ifdef DEBUG_HIGH_LEVEL
      assert(local_id < MAX_NUM_PROCS);
#endif
      if (instances[local_id] == NULL)
        instances[local_id] = new LegionProfInstance(this);
      instances[local_id]->register_task_variant(variant);
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
    void LegionProfiler::register_task(SingleTask *task)
    //--------------------------------------------------------------------------
    {
      Processor current = Processor::get_executing_processor();
      size_t local_id = current.local_id(); 
#ifdef DEBUG_HIGH_LEVEL
      assert(local_id < MAX_NUM_PROCS);
#endif
      if (instances[local_id] == NULL)
        instances[local_id] = new LegionProfInstance(this);
      instances[local_id]->register_task(task);
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_task_request(Realm::ProfilingRequestSet &requests,
                                    Processor::TaskFuncID tid, SingleTask *task)
    //--------------------------------------------------------------------------
    {
      ProfilingInfo info(LEGION_PROF_TASK); 
      info.id = tid;
      info.op_id = task->get_unique_task_id();
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists()) 
                        ? target_proc : Processor::get_executing_processor(),
                        HLR_PROFILING_ID, &info, sizeof(info));
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
      ProfilingInfo info(LEGION_PROF_META); 
      info.id = tid;
      info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists()) 
                        ? target_proc : Processor::get_executing_processor(),
                        HLR_PROFILING_ID, &info, sizeof(info));
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
                        HLR_PROFILING_ID, &info, sizeof(info));
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationTimeline>();
      req.add_measurement<
                Realm::ProfilingMeasurements::OperationMemoryUsage>();
    }

    //--------------------------------------------------------------------------
    void LegionProfiler::add_inst_request(Realm::ProfilingRequestSet &requests,
                                          PhysicalInstance inst, Operation *op)
    //--------------------------------------------------------------------------
    {
      ProfilingInfo info(LEGION_PROF_INST); 
      info.id = inst.id;
      info.op_id = (op != NULL) ? op->get_unique_op_id() : 0;
      Realm::ProfilingRequest &req = requests.add_request((target_proc.exists()) 
                        ? target_proc : Processor::get_executing_processor(),
                        HLR_PROFILING_ID, &info, sizeof(info));
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
            instances[local_id]->process_inst(info->id, info->op_id,
                                              timeline, usage);
            delete timeline;
            delete usage;
            break;
          }
        default:
          assert(false);
      }
    }

  };
};
