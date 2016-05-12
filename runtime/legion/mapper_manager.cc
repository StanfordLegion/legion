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

#include "legion.h"
#include "legion_ops.h"
#include "legion_tasks.h"
#include "mapper_manager.h"
#include "legion_instances.h"
#include "garbage_collection.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Mapping Call Info 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MappingCallInfo::MappingCallInfo(MapperManager *man, MappingCallKind k,
                                     Operation *op /*= NULL*/)
      : manager(man), resume(UserEvent::NO_USER_EVENT), 
        kind(k), operation(op), acquired_instances((op == NULL) ? NULL :
             operation->get_acquired_instances_ref()), start_time(0)
    //--------------------------------------------------------------------------
    {
    }

    /////////////////////////////////////////////////////////////
    // Mapper Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapperManager::MapperManager(Runtime *rt, Mapping::Mapper *mp, 
                                 MapperID mid, Processor p)
      : runtime(rt), mapper(mp), mapper_id(mid), processor(p),
        mapper_lock(Reservation::create_reservation()), next_mapper_event(1)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    MapperManager::~MapperManager(void)
    //--------------------------------------------------------------------------
    {
      // We can now delete our mapper
      delete mapper;
      mapper_lock.destroy_reservation();
      mapper_lock = Reservation::NO_RESERVATION;
    }

    //--------------------------------------------------------------------------
    const char* MapperManager::get_mapper_name(void)
    //--------------------------------------------------------------------------
    {
      return mapper->get_mapper_name();
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_task_options(TaskOp *task, 
     Mapper::TaskOptions *options, bool first_invocation, MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(SELECT_TASK_OPTIONS_CALL,
                            NULL, first_invocation, continuation_precondition);
        // If we need to build a continuation do that now
        if (continuation_precondition.exists())
        {
          MapperContinuation2<TaskOp, Mapper::TaskOptions,
                              &MapperManager::invoke_select_task_options>
                                continuation(this, task, options, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      // If we have an info, we know we are good to go
      mapper->select_task_options(info, *task, *options);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_premap_task(TaskOp *task, 
                                           Mapper::PremapTaskInput *input,
                                           Mapper::PremapTaskOutput *output, 
                                           bool first_invocation,
                                           MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(PREMAP_TASK_CALL,
                             task, first_invocation, continuation_precondition);
        // Build a continuation if necessary
        if (continuation_precondition.exists())
        {
          MapperContinuation3<TaskOp, Mapper::PremapTaskInput, 
            Mapper::PremapTaskOutput, &MapperManager::invoke_premap_task>
              continuation(this, task, input, output, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->premap_task(info, *task, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_slice_task(TaskOp *task, 
                                          Mapper::SliceTaskInput *input,
                                          Mapper::SliceTaskOutput *output, 
                                          bool first_invocation,
                                          MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(SLICE_TASK_CALL,
                             NULL, first_invocation, continuation_precondition);
        // Build a continuation if necessary
        if (continuation_precondition.exists())
        {
          MapperContinuation3<TaskOp, Mapper::SliceTaskInput,
            Mapper::SliceTaskOutput, &MapperManager::invoke_slice_task>
              continuation(this, task, input, output, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->slice_task(info, *task, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_task(TaskOp *task, 
                                        Mapper::MapTaskInput *input,
                                        Mapper::MapTaskOutput *output, 
                                        bool first_invocation,
                                        MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(MAP_TASK_CALL,
                             task, first_invocation, continuation_precondition);
        // Build a continuation if necessary
        if (continuation_precondition.exists())
        {
          MapperContinuation3<TaskOp,Mapper::MapTaskInput,Mapper::MapTaskOutput,
                              &MapperManager::invoke_map_task>
                                continuation(this, task, input, output, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->map_task(info, *task, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_task_variant(TaskOp *task,
                                            Mapper::SelectVariantInput *input,
                                            Mapper::SelectVariantOutput *output,
                                            bool first_invocation,
                                            MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(SELECT_VARIANT_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<TaskOp, Mapper::SelectVariantInput, 
            Mapper::SelectVariantOutput, 
            &MapperManager::invoke_select_task_variant>
              continuation(this, task, input, output, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->select_task_variant(info, *task, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_post_map_task(TaskOp *task, 
                                             Mapper::PostMapInput *input,
                                             Mapper::PostMapOutput *output,
                                             bool first_invocation,
                                             MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(POSTMAP_TASK_CALL,
                             task, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<TaskOp,Mapper::PostMapInput,Mapper::PostMapOutput,
                              &MapperManager::invoke_post_map_task>
                                continuation(this, task, input, output, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->postmap_task(info, *task, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_task_sources(TaskOp *task, 
                                    Mapper::SelectTaskSrcInput *input,
                                    Mapper::SelectTaskSrcOutput *output,
                                    bool first_invocation,
                                    MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(TASK_SELECT_SOURCES_CALL,
                             task, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<TaskOp, Mapper::SelectTaskSrcInput,
            Mapper::SelectTaskSrcOutput, 
            &MapperManager::invoke_select_task_sources>
              continuation(this, task, input, output, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->select_task_sources(info, *task, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_task_speculate(TaskOp *task,
                                              Mapper::SpeculativeOutput *output,
                                              bool first_invocation,
                                              MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(TASK_SPECULATE_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<TaskOp, Mapper::SpeculativeOutput,
                              &MapperManager::invoke_task_speculate>
                                continuation(this, task, output, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->speculate(info, *task, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_task_report_profiling(TaskOp *task, 
                                              Mapper::TaskProfilingInfo *input,
                                              bool first_invocation,
                                              MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(TASK_REPORT_PROFILING_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<TaskOp, Mapper::TaskProfilingInfo,
                              &MapperManager::invoke_task_report_profiling>
                                continuation(this, task, input, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->report_profiling(info, *task, *input);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_inline(MapOp *op, 
                                          Mapper::MapInlineInput *input,
                                          Mapper::MapInlineOutput *output, 
                                          bool first_invocation,
                                          MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(MAP_INLINE_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<MapOp, 
                  Mapper::MapInlineInput,Mapper::MapInlineOutput,
                              &MapperManager::invoke_map_inline>
                                continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->map_inline(info, *op, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_inline_sources(MapOp *op, 
                                      Mapper::SelectInlineSrcInput *input,
                                      Mapper::SelectInlineSrcOutput *output,
                                      bool first_invocation,
                                      MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(INLINE_SELECT_SOURCES_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<MapOp, Mapper::SelectInlineSrcInput,
                              Mapper::SelectInlineSrcOutput, 
                              &MapperManager::invoke_select_inline_sources>
                                continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->select_inline_sources(info, *op, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_inline_report_profiling(MapOp *op, 
                                     Mapper::InlineProfilingInfo *input,
                                     bool first_invocation,
                                     MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(INLINE_REPORT_PROFILING_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<MapOp, Mapper::InlineProfilingInfo,
                              &MapperManager::invoke_inline_report_profiling>
                                continuation(this, op, input, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->report_profiling(info, *op, *input);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_copy(CopyOp *op,
                                        Mapper::MapCopyInput *input,
                                        Mapper::MapCopyOutput *output,
                                        bool first_invocation,
                                        MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(MAP_COPY_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<CopyOp,Mapper::MapCopyInput,Mapper::MapCopyOutput,
                              &MapperManager::invoke_map_copy>
                                continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->map_copy(info, *op, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_copy_sources(CopyOp *op,
                                    Mapper::SelectCopySrcInput *input,
                                    Mapper::SelectCopySrcOutput *output,
                                    bool first_invocation,
                                    MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(COPY_SELECT_SOURCES_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<CopyOp, Mapper::SelectCopySrcInput,
            Mapper::SelectCopySrcOutput, 
            &MapperManager::invoke_select_copy_sources>
              continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->select_copy_sources(info, *op, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_copy_speculate(CopyOp *op, 
                                              Mapper::SpeculativeOutput *output,
                                              bool first_invocation,
                                              MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(COPY_SPECULATE_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<CopyOp, Mapper::SpeculativeOutput,
                              &MapperManager::invoke_copy_speculate>
                                continuation(this, op, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->speculate(info, *op, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_copy_report_profiling(CopyOp *op,
                                             Mapper::CopyProfilingInfo *input,
                                             bool first_invocation,
                                             MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(COPY_REPORT_PROFILING_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<CopyOp, Mapper::CopyProfilingInfo,
                              &MapperManager::invoke_copy_report_profiling>
                                continuation(this, op, input, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->report_profiling(info, *op, *input);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_close(CloseOp *op,
                                         Mapper::MapCloseInput *input,
                                         Mapper::MapCloseOutput *output,
                                         bool first_invocation,
                                         MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(MAP_CLOSE_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<CloseOp, Mapper::MapCloseInput, 
                  Mapper::MapCloseOutput, &MapperManager::invoke_map_close>
                    continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->map_close(info, *op, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_close_sources(CloseOp *op,
                                         Mapper::SelectCloseSrcInput *input,
                                         Mapper::SelectCloseSrcOutput *output,
                                         bool first_invocation,
                                         MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(CLOSE_SELECT_SOURCES_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<CloseOp, Mapper::SelectCloseSrcInput,
            Mapper::SelectCloseSrcOutput, 
            &MapperManager::invoke_select_close_sources>
              continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->select_close_sources(info, *op, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_close_report_profiling(CloseOp *op,
                                          Mapper::CloseProfilingInfo *input,
                                          bool first_invocation,
                                          MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(CLOSE_REPORT_PROFILING_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<CloseOp, Mapper::CloseProfilingInfo,
                              &MapperManager::invoke_close_report_profiling>
                                continuation(this, op, input, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->report_profiling(info, *op, *input);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_acquire(AcquireOp *op,
                                           Mapper::MapAcquireInput *input,
                                           Mapper::MapAcquireOutput *output,
                                           bool first_invocation,
                                           MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(MAP_ACQUIRE_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<AcquireOp, Mapper::MapAcquireInput,
            Mapper::MapAcquireOutput, &MapperManager::invoke_map_acquire>
              continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->map_acquire(info, *op, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_acquire_speculate(AcquireOp *op,
                                             Mapper::SpeculativeOutput *output,
                                             bool first_invocation,
                                             MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(ACQUIRE_SPECULATE_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<AcquireOp, Mapper::SpeculativeOutput,
                              &MapperManager::invoke_acquire_speculate>
                                continuation(this, op, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->speculate(info, *op, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_acquire_report_profiling(AcquireOp *op,
                                         Mapper::AcquireProfilingInfo *input,
                                         bool first_invocation,
                                         MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(ACQUIRE_REPORT_PROFILING_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<AcquireOp, Mapper::AcquireProfilingInfo,
                              &MapperManager::invoke_acquire_report_profiling>
                                continuation(this, op, input, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->report_profiling(info, *op, *input);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_release(ReleaseOp *op,
                                           Mapper::MapReleaseInput *input,
                                           Mapper::MapReleaseOutput *output,
                                           bool first_invocation,
                                           MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(MAP_RELEASE_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<ReleaseOp, Mapper::MapReleaseInput,
            Mapper::MapReleaseOutput, &MapperManager::invoke_map_release>
              continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->map_release(info, *op, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_release_sources(ReleaseOp *op,
                                       Mapper::SelectReleaseSrcInput *input,
                                       Mapper::SelectReleaseSrcOutput *output,
                                       bool first_invocation,
                                       MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(RELEASE_SELECT_SOURCES_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<ReleaseOp, Mapper::SelectReleaseSrcInput,
                              Mapper::SelectReleaseSrcOutput, 
                              &MapperManager::invoke_select_release_sources>
                                continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->select_release_sources(info, *op, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_release_speculate(ReleaseOp *op,
                                             Mapper::SpeculativeOutput *output,
                                             bool first_invocation,
                                             MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(RELEASE_SPECULATE_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<ReleaseOp, Mapper::SpeculativeOutput,
                              &MapperManager::invoke_release_speculate>
                                continuation(this, op, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->speculate(info, *op, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_release_report_profiling(ReleaseOp *op,
                                         Mapper::ReleaseProfilingInfo *input,
                                         bool first_invocation,
                                         MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(RELEASE_REPORT_PROFILING_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<ReleaseOp, Mapper::ReleaseProfilingInfo,
                              &MapperManager::invoke_release_report_profiling>
                                continuation(this, op, input, info);
          continuation.defer(runtime, continuation_precondition, op);  
          return;
        }
      }
      mapper->report_profiling(info, *op, *input);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_configure_context(TaskOp *task,
                                         Mapper::ContextConfigOutput *output,
                                         bool first_invocation,
                                         MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(CONFIGURE_CONTEXT_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<TaskOp, Mapper::ContextConfigOutput,
                              &MapperManager::invoke_configure_context>
                                continuation(this, task, output, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->configure_context(info, *task, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_tunable_value(TaskOp *task,
                                     Mapper::SelectTunableInput *input,
                                     Mapper::SelectTunableOutput *output,
                                     bool first_invocation,
                                     MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(SELECT_TUNABLE_VALUE_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<TaskOp, Mapper::SelectTunableInput,
                              Mapper::SelectTunableOutput, 
                              &MapperManager::invoke_select_tunable_value>
                                continuation(this, task, input, output, info);
          continuation.defer(runtime, continuation_precondition, task);
          return;
        }
      }
      mapper->select_tunable_value(info, *task, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_must_epoch(MustEpochOp *op,
                                            Mapper::MapMustEpochInput *input,
                                            Mapper::MapMustEpochOutput *output,
                                            bool first_invocation,
                                            MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(MAP_MUST_EPOCH_CALL,
                              op, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation3<MustEpochOp, Mapper::MapMustEpochInput, 
                              Mapper::MapMustEpochOutput,
                              &MapperManager::invoke_map_must_epoch>
                                continuation(this, op, input, output, info);
          continuation.defer(runtime, continuation_precondition, op);
          return;
        }
      }
      mapper->map_must_epoch(info, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_dataflow_graph(
                                   Mapper::MapDataflowGraphInput *input,
                                   Mapper::MapDataflowGraphOutput *output,
                                   bool first_invocation,
                                   MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(MAP_DATAFLOW_GRAPH_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<Mapper::MapDataflowGraphInput, 
                              Mapper::MapDataflowGraphOutput,
                              &MapperManager::invoke_map_dataflow_graph>
                                continuation(this, input, output, info);
          continuation.defer(runtime, continuation_precondition);
          return;
        }
      }
      mapper->map_dataflow_graph(info, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_tasks_to_map(
                                    Mapper::SelectMappingInput *input,
                                    Mapper::SelectMappingOutput *output,
                                    bool first_invocation,
                                    MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(SELECT_TASKS_TO_MAP_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<Mapper::SelectMappingInput,
                              Mapper::SelectMappingOutput,
                              &MapperManager::invoke_select_tasks_to_map>
                                continuation(this, input, output, info);
          continuation.defer(runtime, continuation_precondition);
          return;
        }
      }
      mapper->select_tasks_to_map(info, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_steal_targets(
                                     Mapper::SelectStealingInput *input,
                                     Mapper::SelectStealingOutput *output,
                                     bool first_invocation,
                                     MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(SELECT_STEAL_TARGETS_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<Mapper::SelectStealingInput,
                              Mapper::SelectStealingOutput,
                              &MapperManager::invoke_select_steal_targets>
                                continuation(this, input, output, info);
          continuation.defer(runtime, continuation_precondition);
          return;
        }
      }
      mapper->select_steal_targets(info, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_permit_steal_request(
                                     Mapper::StealRequestInput *input,
                                     Mapper::StealRequestOutput *output,
                                     bool first_invocation,
                                     MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(PERMIT_STEAL_REQUEST_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation2<Mapper::StealRequestInput,
                              Mapper::StealRequestOutput,
                              &MapperManager::invoke_permit_steal_request>
                                continuation(this, input, output, info);
          continuation.defer(runtime, continuation_precondition);
          return;
        }
      }
      mapper->permit_steal_request(info, *input, *output);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_handle_message(Mapper::MapperMessage *message,
                                              bool first_invocation,
                                              MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        // Special case for handle message, always defer it if we are also
        // the sender in order to avoid deadlocks
        if ((message->sender == processor) && first_invocation)
        {
          defer_message(message);
          return;
        }
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(HANDLE_MESSAGE_CALL,
                          NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation1<Mapper::MapperMessage,
                              &MapperManager::invoke_handle_message>
                                continuation(this, message, info);
          continuation.defer(runtime, continuation_precondition);
          return;
        }
      }
      mapper->handle_message(info, *message);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_handle_task_result(
                                   Mapper::MapperTaskResult *result,
                                   bool first_invocation,
                                   MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (info == NULL)
      {
        Event continuation_precondition = Event::NO_EVENT;
        info = begin_mapper_call(HANDLE_TASK_RESULT_CALL,
                             NULL, first_invocation, continuation_precondition);
        if (continuation_precondition.exists())
        {
          MapperContinuation1<Mapper::MapperTaskResult,
                              &MapperManager::invoke_handle_task_result>
                                continuation(this, result, info);
          continuation.defer(runtime, continuation_precondition);
          return;
        }
      }
      mapper->handle_task_result(info, *result);
      finish_mapper_call(info);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MapperManager::finish_mapper_call(
                                   const FinishMapperCallContinuationArgs *args)
    //--------------------------------------------------------------------------
    {
      args->manager->finish_mapper_call(args->info, false/*first invocation*/);
    }

    //--------------------------------------------------------------------------
    void MapperManager::send_message(MappingCallInfo *ctx, Processor target,
                                     const void *message, size_t message_size)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->process_mapper_message(target, mapper_id, processor,
                                      message, message_size);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::broadcast(MappingCallInfo *ctx, const void *message,
                                  size_t message_size, int radix)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->process_mapper_broadcast(mapper_id, processor, message,
                                        message_size, radix, 0/*index*/);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    MapperEvent MapperManager::create_mapper_event(MappingCallInfo *ctx)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      MapperEvent result;
      UserEvent event = UserEvent::create_user_event();
      {
        AutoLock m_lock(mapper_lock);
        result.mapper_event_id = next_mapper_event++;
        mapper_events[result.mapper_event_id] = event;
      }
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::has_mapper_event_triggered(MappingCallInfo *ctx,
                                                   MapperEvent event)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool triggered = true;
      Event wait_on = Event::NO_EVENT;
      {
        AutoLock m_lock(mapper_lock, 1, false/*exclusive*/); 
        std::map<unsigned,UserEvent>::const_iterator finder = 
          mapper_events.find(event.mapper_event_id);
        if (finder != mapper_events.end())
          wait_on = finder->second;
      }
      if (wait_on.exists())
        triggered = wait_on.has_triggered();
      resume_mapper_call(ctx);
      return triggered;
    }
    
    //--------------------------------------------------------------------------
    void MapperManager::trigger_mapper_event(MappingCallInfo *ctx, 
                                             MapperEvent event)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock m_lock(mapper_lock);
        std::map<unsigned,UserEvent>::iterator finder = 
          mapper_events.find(event.mapper_event_id);
        if (finder != mapper_events.end())
        {
          to_trigger = finder->second;
          mapper_events.erase(finder);
        }
      }
      if (to_trigger.exists())
        to_trigger.trigger();
      resume_mapper_call(ctx);
    }
    
    //--------------------------------------------------------------------------
    void MapperManager::wait_on_mapper_event(MappingCallInfo *ctx,
                                             MapperEvent event)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Event wait_on = Event::NO_EVENT;
      {
        AutoLock m_lock(mapper_lock);
        std::map<unsigned,UserEvent>::iterator finder = 
          mapper_events.find(event.mapper_event_id);
        if (finder != mapper_events.end())
          wait_on = finder->second;
      }
      if (wait_on.exists())
        wait_on.wait();
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    const ExecutionConstraintSet& MapperManager::find_execution_constraints(
                            MappingCallInfo *ctx, TaskID task_id, VariantID vid)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantImpl *impl = 
        runtime->find_variant_impl(task_id, vid, true/*can fail*/);
      if (impl == NULL)
      {
        log_run.error("Invalid mapper request: mapper %s requested execution "
                      "constraints for variant %ld in mapper call %s, but "
                      "that variant does not exist.", mapper->get_mapper_name(),
                      vid, get_mapper_call_name(ctx->kind));
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME);
      }
      const ExecutionConstraintSet &result = impl->get_execution_constraints();
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    const TaskLayoutConstraintSet& MapperManager::find_task_layout_constraints(
                            MappingCallInfo *ctx, TaskID task_id, VariantID vid)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantImpl *impl = 
        runtime->find_variant_impl(task_id, vid, true/*can fail*/);
      if (impl == NULL)
      {
        log_run.error("Invalid mapper request: mapper %s requested task layout "
                      "constraints for variant %ld in mapper call %s, but "
                      "that variant does not exist.", mapper->get_mapper_name(),
                      vid, get_mapper_call_name(ctx->kind));
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME);
      }
      const TaskLayoutConstraintSet& result = impl->get_layout_constraints();
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    const LayoutConstraintSet& MapperManager::find_layout_constraints(
                             MappingCallInfo *ctx, LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LayoutConstraints *constraints = 
        runtime->find_layout_constraints(layout_id, true/*can fail*/);
      if (constraints == NULL)
      {
        log_run.error("Invalid mapper request: mapper %s requested layout "
                      "constraints for layout ID %ld in mapper call %s, but "
                      "that layout constraint ID is invalid.",
                      mapper->get_mapper_name(), layout_id,
                      get_mapper_call_name(ctx->kind));
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME);
      }
      resume_mapper_call(ctx);
      return *constraints;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID MapperManager::register_layout(MappingCallInfo *ctx,
                                    const LayoutConstraintSet &constraints,
                                    FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LayoutConstraints *cons = runtime->register_layout(handle, constraints);
      resume_mapper_call(ctx);
      return cons->layout_id;
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_layout(MappingCallInfo *ctx, 
                                       LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->release_layout(layout_id);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::do_constraints_conflict(MappingCallInfo *ctx,
                               LayoutConstraintID set1, LayoutConstraintID set2)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LayoutConstraints *c1 = 
        runtime->find_layout_constraints(set1, true/*can fail*/);
      LayoutConstraints *c2 = 
        runtime->find_layout_constraints(set2, true/*can fail*/);
      if ((c1 == NULL) || (c2 == NULL))
      {
        log_run.error("Invalid mapper request: mapper %s passed layout ID %ld "
                      "to conflict test in mapper call %s, but that layout ID "
                      "is invalid.", mapper->get_mapper_name(), 
                      (c1 == NULL) ? set1 : set2, 
                      get_mapper_call_name(ctx->kind));
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME);
      }
      bool result = c1->conflicts(c2);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::do_constraints_entail(MappingCallInfo *ctx,
                           LayoutConstraintID source, LayoutConstraintID target)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LayoutConstraints *c1 = 
        runtime->find_layout_constraints(source, true/*can fail*/);
      LayoutConstraints *c2 = 
        runtime->find_layout_constraints(target, true/*can fail*/);
      if ((c1 == NULL) || (c2 == NULL))
      {
        log_run.error("Invalid mapper request: mapper %s passed layout ID %ld "
                      "to entailment test in mapper call %s, but that layout "
                      "ID is invalid.", mapper->get_mapper_name(), 
                      (c1 == NULL) ? source : target, 
                      get_mapper_call_name(ctx->kind));
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME);
      }
      bool result = c1->entails(c2);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::find_valid_variants(MappingCallInfo *ctx,TaskID task_id,
                                         std::vector<VariantID> &valid_variants,
                                         Processor::Kind kind)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      TaskImpl *task_impl = runtime->find_or_create_task_impl(task_id);
      task_impl->find_valid_variants(valid_variants, kind);
      resume_mapper_call(ctx);
    }
    
    //--------------------------------------------------------------------------
    bool MapperManager::is_leaf_variant(MappingCallInfo *ctx,
                                        TaskID task_id, VariantID variant_id)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantImpl *impl = runtime->find_variant_impl(task_id, variant_id);
      bool result = impl->is_leaf();
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::is_inner_variant(MappingCallInfo *ctx,
                                         TaskID task_id, VariantID variant_id)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantImpl *impl = runtime->find_variant_impl(task_id, variant_id);
      bool result = impl->is_inner();
      resume_mapper_call(ctx);
      return result;
    }
    
    //--------------------------------------------------------------------------
    bool MapperManager::is_idempotent_variant(MappingCallInfo *ctx,
                                           TaskID task_id, VariantID variant_id)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantImpl *impl = runtime->find_variant_impl(task_id, variant_id);
      bool result = impl->is_idempotent();
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::filter_variants(MappingCallInfo *ctx, const Task &task,
            const std::vector<std::vector<MappingInstance> > &chosen_instances,
                                        std::vector<VariantID> &variants)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      std::map<LayoutConstraintID,LayoutConstraints*> layout_cache;
      for (std::vector<VariantID>::iterator var_it = variants.begin();
            var_it != variants.end(); /*nothing*/)
      {
        VariantImpl *impl = runtime->find_variant_impl(task.task_id, *var_it,
                                                       true/*can_fail*/);
        // Not a valid variant
        if (impl == NULL)
        {
          var_it = variants.erase(var_it);
          continue;
        }
        const TaskLayoutConstraintSet &layout_constraints = 
                                        impl->get_layout_constraints();
        bool conflicts = false;
        for (std::multimap<unsigned,LayoutConstraintID>::const_iterator 
              lay_it = layout_constraints.layouts.begin(); 
              lay_it != layout_constraints.layouts.end(); lay_it++)
        {
          LayoutConstraints *constraints;
          std::map<LayoutConstraintID,LayoutConstraints*>::const_iterator
            finder = layout_cache.find(lay_it->second);
          if (finder == layout_cache.end())
          {
            constraints = runtime->find_layout_constraints(lay_it->second);
            layout_cache[lay_it->second] = constraints;
          }
          else
            constraints = finder->second;
          const std::vector<MappingInstance> &instances = 
                                       chosen_instances[lay_it->first];
          for (unsigned idx = 0; idx < instances.size(); idx++)
          {
            PhysicalManager *manager = instances[idx].impl;
            if (manager->conflicts(constraints))
            {
              conflicts = true;
              break;
            }
          }
          if (conflicts)
            break;
        }
        if (conflicts)
          var_it = variants.erase(var_it);
        else
          var_it++;
      }
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::filter_instances(MappingCallInfo *ctx, const Task &task,
                                         VariantID chosen_variant,
                  std::vector<std::vector<MappingInstance> > &chosen_instances,
                  std::vector<std::set<FieldID> > &missing_fields)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      missing_fields.resize(task.regions.size());
      VariantImpl *impl = runtime->find_variant_impl(task.task_id, 
                                                     chosen_variant);
      const TaskLayoutConstraintSet &layout_constraints = 
                                        impl->get_layout_constraints();
      for (unsigned idx = 0; idx < task.regions.size(); idx++)
      {
        if (idx >= chosen_instances.size())
          continue;
        std::vector<MappingInstance> &instances = chosen_instances[idx]; 
        // Iterate over the layout constraints and filter them
        // We know that instance constraints are complete (all dimensions
        // are fully constrainted), therefore we only need to test for conflicts
        for (std::multimap<unsigned,LayoutConstraintID>::const_iterator lay_it =
              layout_constraints.layouts.lower_bound(idx); lay_it != 
              layout_constraints.layouts.upper_bound(idx); lay_it++)
        {
          LayoutConstraints *constraints = 
            runtime->find_layout_constraints(lay_it->second);
          for (std::vector<MappingInstance>::iterator it = 
                instances.begin(); it != instances.end(); /*nothing*/)
          {
            PhysicalManager *manager = it->impl;
            if (manager->conflicts(constraints))
              it = instances.erase(it);
            else
              it++;
          }
          if (instances.empty())
            break;
        }
        // Now figure out which fields are missing
        std::set<FieldID> &missing = missing_fields[idx];
        missing = task.regions[idx].privilege_fields;
        for (std::vector<MappingInstance>::const_iterator it = 
              instances.begin(); it != instances.end(); it++)
        {
          PhysicalManager *manager = it->impl;
          manager->remove_space_fields(missing);
          if (missing.empty())
            break;
        }
      }
      resume_mapper_call(ctx);
    }
    
    //--------------------------------------------------------------------------
    void MapperManager::filter_instances(MappingCallInfo *ctx, const Task &task,
                            unsigned index, VariantID chosen_variant,
                            std::vector<MappingInstance> &instances,
                            std::set<FieldID> &missing_fields)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantImpl *impl = runtime->find_variant_impl(task.task_id, 
                                                     chosen_variant);
      const TaskLayoutConstraintSet &layout_constraints = 
                                        impl->get_layout_constraints();
      // Iterate over the layout constraints and filter them
      // We know that instance constraints are complete (all dimensions
      // are fully constrainted), therefore we only need to test for conflicts
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator lay_it =
            layout_constraints.layouts.lower_bound(index); lay_it != 
            layout_constraints.layouts.upper_bound(index); lay_it++)
      {
        LayoutConstraints *constraints = 
          runtime->find_layout_constraints(lay_it->second);
        for (std::vector<MappingInstance>::iterator it = 
              instances.begin(); it != instances.end(); /*nothing*/)
        {
          PhysicalManager *manager = it->impl;
          if (manager->conflicts(constraints))
            it = instances.erase(it);
          else
            it++;
        }
        if (instances.empty())
          break;
      }
      // Now see which fields we are missing
      missing_fields = task.regions[index].privilege_fields;
      for (std::vector<MappingInstance>::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        PhysicalManager *manager = it->impl;
        manager->remove_space_fields(missing_fields);
        if (missing_fields.empty())
          break;
      }
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, 
                                    bool acquire, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      if (regions.size() > 1)
        check_region_consistency(ctx, "create_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        log_run.warning("Ignoring acquire request to create_physical_instance "
                        "in unsupported mapper call %s in mapper %s", 
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      bool success = runtime->create_physical_instance(target_memory, 
        constraints, regions, result, mapper_id, processor, acquire, priority,
        (ctx->operation == NULL) ? 0 : ctx->operation->get_unique_op_id());
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, true/*created*/);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result,
                                    bool acquire, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      if (regions.size() > 1)
        check_region_consistency(ctx, "create_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        log_run.warning("Ignoring acquire request to create_physical_instance "
                        "in unsupported mapper call %s in mapper %s", 
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      bool success = runtime->create_physical_instance(target_memory, layout_id,
                      regions, result, mapper_id, processor, acquire, priority,
             (ctx->operation == NULL) ? 0 : ctx->operation->get_unique_op_id());
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, true/*created*/);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_or_create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      if (regions.size() > 1)
        check_region_consistency(ctx, "find_or_create_physical_instance", 
                                 regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        log_run.warning("Ignoring acquire request to find_or_create_physical"
                        "_instance in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      bool success = runtime->find_or_create_physical_instance(target_memory,
                  constraints, regions, result, created, mapper_id, processor, 
                  acquire, priority, tight_region_bounds,
                  (ctx->operation == NULL) ? 0 :
                   ctx->operation->get_unique_op_id());
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, created);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_or_create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      if (regions.size() > 1)
        check_region_consistency(ctx, "find_or_create_physical_instance", 
                                 regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        log_run.warning("Ignoring acquire request to find_or_create_physical"
                        "_instance in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      bool success = runtime->find_or_create_physical_instance(target_memory,
                   layout_id, regions, result, created, mapper_id, processor, 
                   acquire, priority, tight_region_bounds,
                   (ctx->operation == NULL) ? 0 : 
                    ctx->operation->get_unique_op_id());
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, created);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_physical_instance(  
                                    MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      if (regions.size() > 1)
        check_region_consistency(ctx, "find_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        log_run.warning("Ignoring acquire request to find_physical_instance "
                        "in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      bool success = runtime->find_physical_instance(target_memory, constraints,
                                 regions, result, acquire, tight_region_bounds);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, false/*created*/);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_physical_instance(  
                                    MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire,
                                    bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      if (regions.size() > 1)
        check_region_consistency(ctx, "find_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        log_run.warning("Ignoring acquire request to find_physical_instance "
                        "in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      bool success = runtime->find_physical_instance(target_memory, layout_id,
                               regions, result, acquire, tight_region_bounds);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, false/*created*/);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    void MapperManager::set_garbage_collection_priority(MappingCallInfo *ctx,
                           const MappingInstance &instance, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      PhysicalManager *manager = instance.impl;
      if (manager->is_virtual_manager())
        return;
      pause_mapper_call(ctx);
      manager->set_garbage_collection_priority(mapper_id, processor, priority);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_instance(MappingCallInfo *ctx,
                                         const MappingInstance &instance)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        log_run.warning("Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      PhysicalManager *manager = instance.impl;
      // virtual instances are easy
      if (manager->is_virtual_manager())
        return true;
      // See if we already acquired it
      if (ctx->acquired_instances->find(manager) !=
          ctx->acquired_instances->end())
        return true;
      pause_mapper_call(ctx);
      if (manager->try_add_base_valid_ref(MAPPING_ACQUIRE_REF,
                                          !manager->is_owner()))
      {
        record_acquired_instance(ctx, manager, false/*created*/);
        resume_mapper_call(ctx);
        return true;
      }
      else if (manager->is_owner())
      {
        resume_mapper_call(ctx);
        return false;
      }
      std::set<PhysicalManager*> instances; 
      instances.insert(manager);
      std::vector<bool> results(1,true);
      Event wait_on = 
        manager->memory_manager->acquire_instances(instances, results);
      if (wait_on.exists())
        wait_on.wait(); // wait for the results to be ready
      bool success = results[0];
      if (success)
        record_acquired_instance(ctx, manager, false/*created*/);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_instances(MappingCallInfo *ctx,
                                  const std::vector<MappingInstance> &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        log_run.warning("Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      // Quick fast path
      if (instances.size() == 1)
        return acquire_instance(ctx, instances[0]);
      pause_mapper_call(ctx);
      // Figure out which instances we need to acquire and sort by memories
      std::map<MemoryManager*,AcquireStatus> acquire_requests;
      bool local_acquired = perform_local_acquires(ctx, instances,
                                                   acquire_requests, NULL);
      if (acquire_requests.empty())
      {
        resume_mapper_call(ctx);
        return local_acquired;
      }
      bool remote_acquired = perform_remote_acquires(ctx, acquire_requests);
      resume_mapper_call(ctx);
      return (local_acquired && remote_acquired);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_and_filter_instances(MappingCallInfo *ctx,
                                        std::vector<MappingInstance> &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        log_run.warning("Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      // Quick fast path
      if (instances.size() == 1)
      {
        bool result = acquire_instance(ctx, instances[0]);
        if (!result)
          instances.clear();
        return result;
      }
      pause_mapper_call(ctx); 
      // Figure out which instances we need to acquire and sort by memories
      std::map<MemoryManager*,AcquireStatus> acquire_requests;
      std::vector<unsigned> to_erase;
      bool local_acquired = perform_local_acquires(ctx, instances,
                                                  acquire_requests, &to_erase);
      // Filter any invalid local instances
      if (!to_erase.empty())
      {
        // Erase from the back
        for (std::vector<unsigned>::const_reverse_iterator it = 
              to_erase.rbegin(); it != to_erase.rend(); it++)
          instances.erase(instances.begin()+(*it)); 
        to_erase.clear();
      }
      if (acquire_requests.empty())
      {
        resume_mapper_call(ctx);
        return local_acquired;
      }
      bool remote_acquired = perform_remote_acquires(ctx, acquire_requests);
      if (!remote_acquired)
      {
        std::map<PhysicalManager*,std::pair<unsigned,bool> > &already_acquired = 
          *(ctx->acquired_instances);
        // Figure out which instances weren't deleted yet
        for (unsigned idx = 0; idx < instances.size(); idx++)
        {
          if (already_acquired.find(instances[idx].impl) == 
              already_acquired.end())
            to_erase.push_back(idx);
        }
        if (!to_erase.empty())
        {
          // Erase from the back
          for (std::vector<unsigned>::const_reverse_iterator it = 
                to_erase.rbegin(); it != to_erase.rend(); it++)
            instances.erase(instances.begin()+(*it));
        }
      }
      return (local_acquired && remote_acquired);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_instances(MappingCallInfo *ctx,
                    const std::vector<std::vector<MappingInstance> > &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        log_run.warning("Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      pause_mapper_call(ctx); 
      // Figure out which instances we need to acquire and sort by memories
      std::map<MemoryManager*,AcquireStatus> acquire_requests;
      bool local_acquired = true;
      for (std::vector<std::vector<MappingInstance> >::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        if (!perform_local_acquires(ctx, *it, acquire_requests, NULL))
          local_acquired = false;
      }
      if (acquire_requests.empty())
      {
        resume_mapper_call(ctx);
        return local_acquired;
      }
      bool remote_acquired = perform_remote_acquires(ctx, acquire_requests);
      resume_mapper_call(ctx);
      return (local_acquired && remote_acquired);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_and_filter_instances(MappingCallInfo *ctx,
                          std::vector<std::vector<MappingInstance> > &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        log_run.warning("Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      pause_mapper_call(ctx);
      // Figure out which instances we need to acquire and sort by memories
      std::map<MemoryManager*,AcquireStatus> acquire_requests;
      std::vector<unsigned> to_erase;
      bool local_acquired = true;
      for (std::vector<std::vector<MappingInstance> >::iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        if (!perform_local_acquires(ctx, *it, acquire_requests, &to_erase))
        {
          local_acquired = false;
          // Erase from the back
          for (std::vector<unsigned>::const_reverse_iterator rit = 
                to_erase.rbegin(); rit != to_erase.rend(); rit++)
            it->erase(it->begin()+(*rit));
          to_erase.clear();
        }
      }
      if (acquire_requests.empty())
      {
        resume_mapper_call(ctx);
        return local_acquired;
      }
      bool remote_acquired = perform_remote_acquires(ctx, acquire_requests);
      if (!remote_acquired)
      {
        std::map<PhysicalManager*,std::pair<unsigned,bool> > &already_acquired =
          *(ctx->acquired_instances); 
        std::vector<unsigned> to_erase;
        for (std::vector<std::vector<MappingInstance> >::iterator it = 
              instances.begin(); it != instances.end(); it++)
        {
          std::vector<MappingInstance> &current = *it;
          for (unsigned idx = 0; idx < current.size(); idx++)
          {
            if (already_acquired.find(current[idx].impl) == 
                already_acquired.end())
              to_erase.push_back(idx);
          }
          if (!to_erase.empty())
          {
            // Erase from the back
            for (std::vector<unsigned>::const_reverse_iterator rit = 
                  to_erase.rbegin(); rit != to_erase.rend(); rit++)
              current.erase(current.begin()+(*rit));
            to_erase.clear();
          }
        }
      }
      resume_mapper_call(ctx);
      return (local_acquired && remote_acquired);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::perform_local_acquires(MappingCallInfo *info,
                      const std::vector<MappingInstance> &instances,
                      std::map<MemoryManager*,AcquireStatus> &acquire_requests,
                      std::vector<unsigned> *to_erase)
    //--------------------------------------------------------------------------
    {
      std::map<PhysicalManager*,std::pair<unsigned,bool> > &already_acquired = 
        *(info->acquired_instances);
      bool local_acquired = true;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        PhysicalManager *manager = instances[idx].impl;
        if (manager->is_virtual_manager())
          continue;
        if (already_acquired.find(manager) != already_acquired.end())
          continue;
        // Try to add an acquired reference immediately
        // If we're remote it has to be valid already to be sound, but if
        // we're local whatever works
        if (manager->try_add_base_valid_ref(MAPPING_ACQUIRE_REF, 
                                            !manager->is_owner()))
        {
          // We already know it wasn't there before
          already_acquired[manager] = std::pair<unsigned,bool>(1, false);
          continue;
        }
        // if we failed on the owner node, it will never work
        else if (manager->is_owner()) 
        {
          if (to_erase != NULL)
            to_erase->push_back(idx);
          local_acquired = false;
          continue; 
        }
        acquire_requests[manager->memory_manager].instances.insert(manager);
      }
      return local_acquired;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::perform_remote_acquires(MappingCallInfo *info,
                       std::map<MemoryManager*,AcquireStatus> &acquire_requests)
    //--------------------------------------------------------------------------
    {
      std::set<Event> done_events;
      // Issue the requests and see what we need to wait on
      for (std::map<MemoryManager*,AcquireStatus>::iterator it = 
            acquire_requests.begin(); it != acquire_requests.end(); it++)
      {
        Event wait_on = it->first->acquire_instances(it->second.instances,
                                                     it->second.results);
        if (wait_on.exists())
          done_events.insert(wait_on);          
      }
      // See if we have to wait for our results to be done
      if (!done_events.empty())
      {
        Event ready = Runtime::merge_events<true>(done_events);
        ready.wait();
      }
      // Now find out which ones we acquired and which ones didn't
      bool all_acquired = true;
      for (std::map<MemoryManager*,AcquireStatus>::const_iterator req_it = 
            acquire_requests.begin(); req_it != acquire_requests.end();req_it++)
      {
        unsigned idx = 0;
        for (std::set<PhysicalManager*>::const_iterator it =  
              req_it->second.instances.begin(); it != 
              req_it->second.instances.end(); it++, idx++)
        {
          if (req_it->second.results[idx])
          {
            // record that we acquired it
            record_acquired_instance(info, *it, false/*created*/); 
            // make the reference a local reference and 
            // remove our remote did reference
            (*it)->add_base_valid_ref(MAPPING_ACQUIRE_REF);
            (*it)->send_remote_valid_update(req_it->first->owner_space,
                                            1, false/*add*/);
          }
          else
            all_acquired = false;
        }
      }
      return all_acquired;
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_instance(MappingCallInfo *ctx, 
                                         const MappingInstance &instance)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        log_run.warning("Ignoring release request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return;
      }
      pause_mapper_call(ctx);
      release_acquired_instance(ctx, instance.impl); 
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_instances(MappingCallInfo *ctx,
                                 const std::vector<MappingInstance> &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        log_run.warning("Ignoring release request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return;
      }
      pause_mapper_call(ctx);
      for (unsigned idx = 0; idx < instances.size(); idx++)
        release_acquired_instance(ctx, instances[idx].impl);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_instances(MappingCallInfo *ctx, 
                   const std::vector<std::vector<MappingInstance> > &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        log_run.warning("Ignoring release request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return;
      }
      pause_mapper_call(ctx);
      for (std::vector<std::vector<MappingInstance> >::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        for (unsigned idx = 0; idx < it->size(); idx++)
          release_acquired_instance(ctx, (*it)[idx].impl);
      }
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::record_acquired_instance(MappingCallInfo *ctx,
                                         PhysicalManager *manager, bool created)
    //--------------------------------------------------------------------------
    {
      if (manager->is_virtual_manager())
        return;
#ifdef DEBUG_LEGION
      assert(ctx->acquired_instances != NULL);
#endif
      std::map<PhysicalManager*,
        std::pair<unsigned,bool> > &acquired =*(ctx->acquired_instances);
      std::map<PhysicalManager*,std::pair<unsigned,bool> >::iterator finder = 
        acquired.find(manager); 
      if (finder == acquired.end())
        acquired[manager] = std::pair<unsigned,bool>(1/*first ref*/, created);
      else
        finder->second.first++;
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_acquired_instance(MappingCallInfo *ctx,
                                                  PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      if (manager->is_virtual_manager())
        return;
#ifdef DEBUG_LEGION
      assert(ctx->acquired_instances != NULL);
#endif
      std::map<PhysicalManager*,std::pair<unsigned,bool> > &acquired =
        *(ctx->acquired_instances);
      std::map<PhysicalManager*,std::pair<unsigned,bool> >::iterator finder = 
        acquired.find(manager);
      if (finder == acquired.end())
        return;
      // Release the refrences and then keep going, we know there is 
      // a resource reference so no need to check for deletion
      manager->remove_base_valid_ref(MAPPING_ACQUIRE_REF, finder->second.first);
      acquired.erase(finder);
    }

    //--------------------------------------------------------------------------
    void MapperManager::check_region_consistency(MappingCallInfo *info,
                                                 const char *call_name,
                                      const std::vector<LogicalRegion> &regions)
    //--------------------------------------------------------------------------
    {
      RegionTreeID tree_id = 0;
      for (unsigned idx = 0; idx < regions.size(); idx++)
      {
        if (idx > 0)
        {
          RegionTreeID other_id = regions[idx].get_tree_id();
          if (other_id != tree_id)
          {
            log_run.error("Invalid region arguments passed to %s in "
                          "mapper call %s of mapper %s. All region arguments "
                          "must be from the same region tree (%d != %d).",
                          call_name, get_mapper_call_name(info->kind),
                          mapper->get_mapper_name(), tree_id, other_id);
#ifdef DEBUG_LEGION
            assert(false);
#endif
            exit(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME);
          }
        }
        else
          tree_id = regions[idx].get_tree_id();
      }
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperManager::get_index_partition(MappingCallInfo *ctx,
                                                      IndexSpace parent, 
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexPartition result = runtime->get_index_partition(parent, color);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_index_subspace(MappingCallInfo *ctx,
                                                 IndexPartition p, Color c)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexSpace result = runtime->get_index_subspace(p, c);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_index_subspace(MappingCallInfo *ctx,
                                                 IndexPartition p, 
                                                 const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexSpace result = runtime->get_index_subspace(p, color);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::has_multiple_domains(MappingCallInfo *ctx,
                                             IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool result = runtime->has_multiple_domains(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    Domain MapperManager::get_index_space_domain(MappingCallInfo *ctx,
                                                 IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Domain result = runtime->get_index_space_domain(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::get_index_space_domains(MappingCallInfo *ctx,
                                                IndexSpace handle,
                                                std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->get_index_space_domains(handle, domains);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    Domain MapperManager::get_index_partition_color_space(MappingCallInfo *ctx,
                                                          IndexPartition p)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Domain result = runtime->get_index_partition_color_space(p);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::get_index_space_partition_colors(MappingCallInfo *ctx,
                                     IndexSpace handle, std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->get_index_space_partition_colors(handle, colors);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::is_index_partition_disjoint(MappingCallInfo *ctx,
                                                    IndexPartition p)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool result = runtime->is_index_partition_disjoint(p);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_index_space_color(MappingCallInfo *ctx,
                                               IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Color result = runtime->get_index_space_color(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_index_partition_color(MappingCallInfo *ctx,
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Color result = runtime->get_index_partition_color(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_parent_index_space(MappingCallInfo *ctx,
                                                     IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexSpace result = runtime->get_parent_index_space(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::has_parent_index_partition(MappingCallInfo *ctx,
                                                   IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool result = runtime->has_parent_index_partition(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperManager::get_parent_index_partition(
                                        MappingCallInfo *ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexPartition result = runtime->get_parent_index_partition(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    size_t MapperManager::get_field_size(MappingCallInfo *ctx,
                                         FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      size_t result = runtime->get_field_size(handle, fid);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::get_field_space_fields(MappingCallInfo *ctx,
                                FieldSpace handle, std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->get_field_space_fields(handle, fields);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_logical_partition(MappingCallInfo *ctx,
                                                          LogicalRegion parent,
                                                          IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalPartition result = runtime->get_logical_partition(parent, handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_logical_partition_by_color(
                           MappingCallInfo *ctx, LogicalRegion par, Color color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalPartition result = 
        runtime->get_logical_partition_by_color(par, color);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_logical_partition_by_tree(
                                                        MappingCallInfo *ctx,
                                                        IndexPartition part,
                                                        FieldSpace fspace, 
                                                        RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalPartition result = 
        runtime->get_logical_partition_by_tree(part, fspace, tid);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_logical_subregion(MappingCallInfo *ctx,
                                                       LogicalPartition parent,
                                                       IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalRegion result = runtime->get_logical_subregion(parent, handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_logical_subregion_by_color(
                        MappingCallInfo *ctx, LogicalPartition par, Color color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalRegion result = runtime->get_logical_subregion_by_color(par,color);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_logical_subregion_by_tree(
                                      MappingCallInfo *ctx, IndexSpace handle, 
                                      FieldSpace fspace, RegionTreeID tid)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalRegion result = 
        runtime->get_logical_subregion_by_tree(handle, fspace, tid);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_logical_region_color(MappingCallInfo *ctx,
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Color result = runtime->get_logical_region_color(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_logical_partition_color(MappingCallInfo *ctx,
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Color result = runtime->get_logical_partition_color(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_parent_logical_region(MappingCallInfo *ctx,
                                                          LogicalPartition part)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalRegion result = runtime->get_parent_logical_region(part);
      resume_mapper_call(ctx);
      return result;
    }
    
    //--------------------------------------------------------------------------
    bool MapperManager::has_parent_logical_partition(MappingCallInfo *ctx,
                                                     LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool result = runtime->has_parent_logical_partition(handle);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_parent_logical_partition(
                                          MappingCallInfo *ctx, LogicalRegion r)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalPartition result = runtime->get_parent_logical_partition(r);
      resume_mapper_call(ctx);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        TaskID task_id, SemanticTag tag, const void *&result, size_t &size,
        bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->retrieve_semantic_information(task_id, tag, result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        IndexSpace handle, SemanticTag tag, const void *&result, size_t &size,
        bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->retrieve_semantic_information(handle, tag, result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        IndexPartition handle, SemanticTag tag, const void *&result, 
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->retrieve_semantic_information(handle, tag, result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        FieldSpace handle, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->retrieve_semantic_information(handle, tag, result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        FieldSpace handle, FieldID fid, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->retrieve_semantic_information(handle, fid, tag, result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        LogicalRegion handle, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->retrieve_semantic_information(handle, tag, result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        LogicalPartition handle, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->retrieve_semantic_information(handle, tag, result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, TaskID task_id,
                                      const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(task_id, NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      result = reinterpret_cast<const char*>(name);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, IndexSpace handle,
                                      const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      result = reinterpret_cast<const char*>(name);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, 
                                      IndexPartition handle,const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      result = reinterpret_cast<const char*>(name);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, FieldSpace handle,
                                      const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      result = reinterpret_cast<const char*>(name);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, FieldSpace handle,
                                      FieldID fid, const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, fid, NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      result = reinterpret_cast<const char*>(name);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, 
                                      LogicalRegion handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      result = reinterpret_cast<const char*>(name);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx,
                                   LogicalPartition handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      result = reinterpret_cast<const char*>(name);
      resume_mapper_call(ctx);
    }

    //--------------------------------------------------------------------------
    MappingCallInfo* MapperManager::allocate_call_info(MappingCallKind kind,
                                                  Operation *op, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock m_lock(mapper_lock);
        return allocate_call_info(kind, op, false/*need lock*/);
      }
      if (!available_infos.empty())
      {
        MappingCallInfo *result = available_infos.back();
        available_infos.pop_back();
        result->kind = kind;
        result->operation = op;
        if (op != NULL)
          result->acquired_instances = op->get_acquired_instances_ref();
        if (runtime->profiler != NULL)
          result->start_time = Realm::Clock::current_time_in_nanoseconds();
        return result;
      }
      MappingCallInfo *result = new MappingCallInfo(this, kind, op);
      if (runtime->profiler != NULL)
        result->start_time = Realm::Clock::current_time_in_nanoseconds();
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::free_call_info(MappingCallInfo *info, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock m_lock(mapper_lock);
        free_call_info(info, false/*need lock*/);
        return;
      }
      if (runtime->profiler != NULL)
      {
        unsigned long long stop_time = 
          Realm::Clock::current_time_in_nanoseconds();
        runtime->profiler->record_mapper_call(info->kind, 
            (info->operation == NULL) ? 0 : info->operation->get_unique_op_id(),
            info->start_time, stop_time); 
      }
      info->resume = UserEvent::NO_USER_EVENT;
      info->operation = NULL;
      info->acquired_instances = NULL;
      info->start_time = 0;
      available_infos.push_back(info);
    }

    //--------------------------------------------------------------------------
    /*static*/ const char* MapperManager::get_mapper_call_name(
                                                           MappingCallKind kind)
    //--------------------------------------------------------------------------
    {
      MAPPER_CALL_NAMES(call_names); 
#ifdef DEBUG_LEGION
      assert(kind < LAST_MAPPER_CALL);
#endif
      return call_names[kind];
    }

    //--------------------------------------------------------------------------
    void MapperManager::defer_message(Mapper::MapperMessage *message)
    //--------------------------------------------------------------------------
    {
      // Acquire the lock as the precondition
      Event precondition = mapper_lock.acquire(0, true/*exclusive*/);
      DeferMessageArgs args;
      args.hlr_id = HLR_DEFER_MAPPER_MESSAGE_TASK_ID;
      args.manager = this;
      args.sender = message->sender;
      args.size = message->size;
      args.message = malloc(args.size);
      memcpy(args.message, message->message, args.size);
      args.broadcast = message->broadcast;
      runtime->issue_runtime_meta_task(&args, sizeof(args), 
          HLR_DEFER_MAPPER_MESSAGE_TASK_ID, HLR_RESOURCE_PRIORITY,
          NULL, precondition);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MapperManager::handle_deferred_message(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMessageArgs *margs = (const DeferMessageArgs*)args;
      Mapper::MapperMessage message;
      message.sender = margs->sender;
      message.message = margs->message;
      message.size = margs->size;
      message.broadcast = margs->broadcast;
      margs->manager->invoke_handle_message(&message,false/*first invocation*/);
      // Then free up the allocated memory
      free(margs->message);
    }

    /////////////////////////////////////////////////////////////
    // Serializing Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SerializingManager::SerializingManager(Runtime *rt, Mapping::Mapper *mp,
                             MapperID map_id, Processor p, bool init_reentrant)
      : MapperManager(rt, mp, map_id, p), permit_reentrant(init_reentrant), 
        executing_call(NULL), paused_calls(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SerializingManager::SerializingManager(const SerializingManager &rhs)
      : MapperManager(NULL,NULL,0,Processor::NO_PROC)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    SerializingManager::~SerializingManager(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    SerializingManager& SerializingManager::operator=(
                                                  const SerializingManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool SerializingManager::is_locked(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Serializing managers are always effectively locked
      return true;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::lock_mapper(MappingCallInfo *info, bool read_only)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void SerializingManager::unlock_mapper(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    bool SerializingManager::is_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(executing_call == info);
#endif
      // No need to hold the lock here since we are exclusive
      return permit_reentrant;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::enable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(executing_call == info);
#endif
      // No need to hold the lock since we know we are exclusive 
      permit_reentrant = true;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::disable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(executing_call == info);
#endif
      // No need to hold the lock since we know we are exclusive
      if (permit_reentrant)
      {
        // If there are paused calls, we need to wait for them all 
        // to finish before we can continue execution 
        if (paused_calls > 0)
        {
#ifdef DEBUG_LEGION
          assert(!info->resume.exists());
#endif
          UserEvent ready_event = UserEvent::create_user_event();
          info->resume = ready_event;
          non_reentrant_calls.push_back(info);
          ready_event.wait();
          // When we wake up, we should be non-reentrant
#ifdef DEBUG_LEGION
          assert(!permit_reentrant);
#endif
        }
        else
          permit_reentrant = false;
      }
    }

    //--------------------------------------------------------------------------
    MappingCallInfo* SerializingManager::begin_mapper_call(MappingCallKind kind,
                      Operation *op, bool first_invocation, Event &precondition)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo *result = NULL;
      // If this is the first invocation we have to ask for the lock
      // otherwise we know we already have it
      if (first_invocation)
      {
        precondition = mapper_lock.acquire(0, true/*exclusive*/);
        if (!precondition.has_triggered())
          return NULL;
      }
      result = allocate_call_info(kind, op, false/*need lock*/);
      // See if we are ready to run this or not
      if ((executing_call != NULL) || (!permit_reentrant && 
            ((paused_calls > 0) || !ready_calls.empty())))
      {
        // Put this on the list of pending calls
        result->resume = UserEvent::create_user_event();
        precondition = result->resume;
        pending_calls.push_back(result);
      }
      else
        executing_call = result;
      mapper_lock.release();
      return result;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::pause_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (executing_call != info)
      {
        log_run.error("ERROR: Invalid mapper context passed to mapper_rt "
                      "call by mapper %s. Mapper contexts are only valid "
                      "for the mapper call to which they are passed. They "
                      "cannot be stored beyond the lifetime of the "
                      "mapper call.", mapper->get_mapper_name());
#ifdef DEBUG_LEGION
        assert(false);
#endif
        exit(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME);
      }
      // We definitely know we can't start any non_reentrant calls
      // Screw fairness, we care about throughput, see if there are any
      // pending calls to wake up, and then go to sleep ourself
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock m_lock(mapper_lock);
        // Increment the count of the paused mapper calls
        paused_calls++;
        if (permit_reentrant && !ready_calls.empty())
        {
          // Get the next ready call to continue executing
          executing_call = ready_calls.front();
          ready_calls.pop_front();
          to_trigger = executing_call->resume;
        }
        else if (permit_reentrant && !pending_calls.empty())
        {
          // Get the next available call to handle
          executing_call = pending_calls.front();
          pending_calls.pop_front();
          to_trigger = executing_call->resume; 
        }
        else // No one to wake up
          executing_call = NULL;
      }
      if (to_trigger.exists())
        to_trigger.trigger();
    }

    //--------------------------------------------------------------------------
    void SerializingManager::resume_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // See if we are ready to be woken up
      Event wait_on = Event::NO_EVENT;
      {
        AutoLock m_lock(mapper_lock);
#ifdef DEBUG_LEGION
        assert(paused_calls > 0);
#endif
        paused_calls--;
        if (executing_call != NULL)
        {
          info->resume = UserEvent::create_user_event();
          wait_on = info->resume;
          ready_calls.push_back(info);
        }
        else
          executing_call = info;
      }
      if (wait_on.exists())
        wait_on.wait();
#ifdef DEBUG_LEGION
      assert(executing_call == info);
#endif
    }

    //--------------------------------------------------------------------------
    void SerializingManager::finish_mapper_call(MappingCallInfo *info,
                                                bool first_invocation)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(executing_call == info);
#endif
      if (first_invocation)
      {
        Event precondition = mapper_lock.acquire();
        if (!precondition.has_triggered())
        {
          FinishMapperCallContinuationArgs args;
          args.hlr_id = HLR_FINISH_MAPPER_CONTINUATION_TASK_ID;
          args.manager = this;
          args.info = info;
          runtime->issue_runtime_meta_task(&args, sizeof(args),
              HLR_FINISH_MAPPER_CONTINUATION_TASK_ID, HLR_RESOURCE_PRIORITY,
              info->operation, precondition);
          // No need to wait, we know the mapper call is done
          // This is all just clean up and can be done asynchronously
          return;
        }
      }
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      // See if can start a non-reentrant task
      if (!non_reentrant_calls.empty() && 
          (paused_calls == 0) && ready_calls.empty())
      {
        // Mark that we are now not permitting re-entrant
        permit_reentrant = false;
        executing_call = non_reentrant_calls.front();
        non_reentrant_calls.pop_front();
        to_trigger = executing_call->resume;
      }
      else if (!ready_calls.empty())
      {
        executing_call = ready_calls.front();
        ready_calls.pop_front();
        to_trigger = executing_call->resume;
      }
      else if (!pending_calls.empty())
      {
        executing_call = pending_calls.front();
        pending_calls.pop_front();
        to_trigger = executing_call->resume;
      }
      else
        executing_call = NULL;
      // Return our call info
      free_call_info(info, false/*need lock*/);
      mapper_lock.release();
      // Wake up the next task if necessary
      if (to_trigger.exists())
        to_trigger.trigger();
    }

    /////////////////////////////////////////////////////////////
    // Concurrent Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ConcurrentManager::ConcurrentManager(Runtime *rt, Mapping::Mapper *mp,
                                         MapperID map_id, Processor p)
      : MapperManager(rt, mp, map_id, p), lock_state(UNLOCKED_STATE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ConcurrentManager::ConcurrentManager(const ConcurrentManager &rhs)
      : MapperManager(NULL,NULL,0,Processor::NO_PROC)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ConcurrentManager::~ConcurrentManager(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ConcurrentManager& ConcurrentManager::operator=(
                                                   const ConcurrentManager &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    bool ConcurrentManager::is_locked(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Can read this without holding the lock
      return (lock_state != UNLOCKED_STATE);  
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::lock_mapper(MappingCallInfo *info, bool read_only)
    //--------------------------------------------------------------------------
    {
      Event wait_on = Event::NO_EVENT;
      {
        AutoLock m_lock(mapper_lock); 
        if (current_holders.find(info) != current_holders.end())
        {
          log_run.error("Invalid duplicate mapper lock request in mapper call "
                        "%s for mapper %s", get_mapper_call_name(info->kind),
                        mapper->get_mapper_name());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_SYNCHRONIZATION);
        }
        switch (lock_state)
        {
          case UNLOCKED_STATE:
            {
              // Grant the lock immediately
              current_holders.insert(info);
              if (read_only)
                lock_state = READ_ONLY_STATE;
              else
                lock_state = EXCLUSIVE_STATE;
              break;
            }
          case READ_ONLY_STATE:
            {
              if (!read_only)
              {
                info->resume = UserEvent::create_user_event();
                wait_on = info->resume;
                exclusive_waiters.push_back(info);
              }
              else // add it to the set of current holders
                current_holders.insert(info);
              break;
            }
          case EXCLUSIVE_STATE:
            {
              // Have to wait no matter what
              info->resume = UserEvent::create_user_event();
              wait_on = info->resume;
              if (read_only)
                read_only_waiters.push_back(info);
              else
                exclusive_waiters.push_back(info);
              break;
            }
          default:
            assert(false);
        }
      }
      if (wait_on.exists())
        wait_on.wait();
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::unlock_mapper(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      std::vector<UserEvent> to_trigger;
      {
        AutoLock m_lock(mapper_lock);
        std::set<MappingCallInfo*>::iterator finder = 
          current_holders.find(info);
        if (finder == current_holders.end())
        {
          log_run.error("Invalid unlock mapper call with no prior lock call "
                        "in mapper call %s for mapper %s",
                        get_mapper_call_name(info->kind),
                        mapper->get_mapper_name());
#ifdef DEBUG_LEGION
          assert(false);
#endif
          exit(ERROR_INVALID_MAPPER_SYNCHRONIZATION);
        }
        current_holders.erase(finder);
        // See if we can now give the lock to someone else
        if (current_holders.empty())
          release_lock(to_trigger);
      }
      if (!to_trigger.empty())
      {
        for (std::vector<UserEvent>::const_iterator it = 
              to_trigger.begin(); it != to_trigger.end(); it++)
          it->trigger();
      }
    }

    //--------------------------------------------------------------------------
    bool ConcurrentManager::is_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Always reentrant for the concurrent manager
      return true;
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::enable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do 
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::disable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    MappingCallInfo* ConcurrentManager::begin_mapper_call(MappingCallKind kind,
                      Operation *op, bool first_invocation, Event &precondition)
    //--------------------------------------------------------------------------
    {
      if (first_invocation)
      {
        precondition = mapper_lock.acquire(0, true/*exclusive*/);
        if (!precondition.has_triggered())
          return NULL;
      }
      MappingCallInfo *result = allocate_call_info(kind, op,false/*need lock*/);
      mapper_lock.release();
      return result;
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::pause_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::resume_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Nothing to do
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::finish_mapper_call(MappingCallInfo *info,
                                               bool first_invocation)
    //--------------------------------------------------------------------------
    {
      if (first_invocation)
      {
        Event precondition = mapper_lock.acquire();
        if (!precondition.has_triggered())
        {
          FinishMapperCallContinuationArgs args;
          args.hlr_id = HLR_FINISH_MAPPER_CONTINUATION_TASK_ID;
          args.manager = this;
          args.info = info;
          runtime->issue_runtime_meta_task(&args, sizeof(args),
              HLR_FINISH_MAPPER_CONTINUATION_TASK_ID, HLR_RESOURCE_PRIORITY,
              info->operation, precondition);
          // No need to wait, we know the mapper call is done
          // This is all just clean up and can be done asynchronously
          return;
        }
      }
      std::vector<UserEvent> to_trigger;
      // Check to see if we need to release the lock for the mapper call
      std::set<MappingCallInfo*>::iterator finder = 
          current_holders.find(info);     
      if (finder != current_holders.end())
      {
        current_holders.erase(finder);
        release_lock(to_trigger);
      }
      free_call_info(info, false/*need lock*/);
      mapper_lock.release();
      if (!to_trigger.empty())
      {
        for (std::vector<UserEvent>::const_iterator it = 
              to_trigger.begin(); it != to_trigger.end(); it++)
          it->trigger();
      }
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::release_lock(std::vector<UserEvent> &to_trigger)
    //--------------------------------------------------------------------------
    {
      switch (lock_state)
      {
        case READ_ONLY_STATE:
          {
            if (!exclusive_waiters.empty())
            {
              // Pull off the first exlusive waiter
              to_trigger.push_back(exclusive_waiters.front()->resume);
              exclusive_waiters.pop_front();
              lock_state = EXCLUSIVE_STATE;
            }
            else
              lock_state = UNLOCKED_STATE;
            break;
          }
        case EXCLUSIVE_STATE:
          {
            if (!read_only_waiters.empty())
            {
              to_trigger.resize(read_only_waiters.size());
              for (unsigned idx = 0; idx < read_only_waiters.size(); idx++)
                to_trigger[idx] = read_only_waiters[idx]->resume;
              read_only_waiters.clear();
              lock_state = READ_ONLY_STATE;
            }
            else
              lock_state = UNLOCKED_STATE;
            break;
          }
        default:
          assert(false);
      }
    }

    /////////////////////////////////////////////////////////////
    // Mapper Continuation 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapperContinuation::MapperContinuation(MapperManager *man,
                                           MappingCallInfo *i)
      : manager(man), info(i)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void MapperContinuation::defer(Runtime *runtime, Event precondition, 
                                   Operation *op)
    //--------------------------------------------------------------------------
    {
      ContinuationArgs args;
      args.hlr_id = HLR_MAPPER_CONTINUATION_TASK_ID;
      args.continuation = this;
      Event wait_on = runtime->issue_runtime_meta_task(&args, sizeof(args),
                          HLR_MAPPER_CONTINUATION_TASK_ID, 
                          HLR_LATENCY_PRIORITY, op, precondition);
      wait_on.wait();
    }

    //--------------------------------------------------------------------------
    /*static*/ void MapperContinuation::handle_continuation(const void *args)
    //--------------------------------------------------------------------------
    {
      const ContinuationArgs *conargs = (const ContinuationArgs*)args;
      conargs->continuation->execute();
    }

  };
}; // namespace Legion
