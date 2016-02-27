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
    // Mapper Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapperManager::MapperManager(Runtime *rt, Mapping::Mapper *mp, 
                                 MapperID mid, Processor p)
      : runtime(rt), mapper(mp), mapper_id(mid), processor(p),
        mapper_lock(Reservation::create_reservation())
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
                            Mapper::TaskOptions *options, bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(SELECT_TASK_OPTIONS_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_task_options(info, *task, *options);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation); // better only get here the first time
#endif
      // Otherwise make a continuation and launch it
      MapperContinuation2<TaskOp, Mapper::TaskOptions,
                          &MapperManager::invoke_select_task_options>
                            continuation(this, task, options);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_premap_task(TaskOp *task, 
                                           Mapper::PremapTaskInput *input,
                                           Mapper::PremapTaskOutput *output, 
                                           bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(PREMAP_TASK_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->premap_task(info, *task, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<TaskOp, Mapper::PremapTaskInput, 
        Mapper::PremapTaskOutput, &MapperManager::invoke_premap_task>
          continuation(this, task, input, output);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_slice_task(TaskOp *task, 
                                          Mapper::SliceTaskInput *input,
                                          Mapper::SliceTaskOutput *output, 
                                          bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(SLICE_TASK_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->slice_task(info, *task, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<TaskOp, Mapper::SliceTaskInput,
        Mapper::SliceTaskOutput, &MapperManager::invoke_slice_task>
          continuation(this, task, input, output);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_task(TaskOp *task, 
                                        Mapper::MapTaskInput *input,
                                        Mapper::MapTaskOutput *output, 
                                        bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(MAP_TASK_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->map_task(info, *task, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<TaskOp, Mapper::MapTaskInput, Mapper::MapTaskOutput,
                          &MapperManager::invoke_map_task>
                            continuation(this, task, input, output);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_task_variant(TaskOp *task,
                                            Mapper::SelectVariantInput *input,
                                            Mapper::SelectVariantOutput *output,
                                            bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(SELECT_VARIANT_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_task_variant(info, *task, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<TaskOp, Mapper::SelectVariantInput, 
        Mapper::SelectVariantOutput, &MapperManager::invoke_select_task_variant>
          continuation(this, task, input, output);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_post_map_task(TaskOp *task, 
                                             Mapper::PostMapInput *input,
                                             Mapper::PostMapOutput *output,
                                             bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(POSTMAP_TASK_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->postmap_task(info, *task, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<TaskOp, Mapper::PostMapInput, Mapper::PostMapOutput,
                          &MapperManager::invoke_post_map_task>
                            continuation(this, task, input, output);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_task_sources(TaskOp *task, 
                                    Mapper::SelectTaskSrcInput *input,
                                    Mapper::SelectTaskSrcOutput *output,
                                    bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(TASK_SELECT_SOURCES_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_task_sources(info, *task, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<TaskOp, Mapper::SelectTaskSrcInput,
        Mapper::SelectTaskSrcOutput, &MapperManager::invoke_select_task_sources>
          continuation(this, task, input, output);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_task_speculate(TaskOp *task,
                                              Mapper::SpeculativeOutput *output,
                                              bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(TASK_SPECULATE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->speculate(info, *task, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<TaskOp, Mapper::SpeculativeOutput,
                          &MapperManager::invoke_task_speculate>
                            continuation(this, task, output);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_task_report_profiling(TaskOp *task, 
                                              Mapper::TaskProfilingInfo *input,
                                              bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(TASK_REPORT_PROFILING_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->report_profiling(info, *task, *input);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<TaskOp, Mapper::TaskProfilingInfo,
                          &MapperManager::invoke_task_report_profiling>
                            continuation(this, task, input);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_inline(MapOp *op, 
                                          Mapper::MapInlineInput *input,
                                          Mapper::MapInlineOutput *output, 
                                          bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(MAP_INLINE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->map_inline(info, *op, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<MapOp, Mapper::MapInlineInput,Mapper::MapInlineOutput,
                          &MapperManager::invoke_map_inline>
                            continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_inline_sources(MapOp *op, 
                                      Mapper::SelectInlineSrcInput *input,
                                      Mapper::SelectInlineSrcOutput *output,
                                      bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(INLINE_SELECT_SOURCES_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_inline_sources(info, *op, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<MapOp, Mapper::SelectInlineSrcInput,
                          Mapper::SelectInlineSrcOutput, 
                          &MapperManager::invoke_select_inline_sources>
                            continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_inline_report_profiling(MapOp *op, 
                                     Mapper::InlineProfilingInfo *input,
                                     bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(INLINE_REPORT_PROFILING_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->report_profiling(info, *op, *input);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<MapOp, Mapper::InlineProfilingInfo,
                          &MapperManager::invoke_inline_report_profiling>
                            continuation(this, op, input);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_copy(CopyOp *op,
                                        Mapper::MapCopyInput *input,
                                        Mapper::MapCopyOutput *output,
                                        bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(MAP_COPY_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->map_copy(info, *op, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<CopyOp, Mapper::MapCopyInput, Mapper::MapCopyOutput,
                          &MapperManager::invoke_map_copy>
                            continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_copy_sources(CopyOp *op,
                                    Mapper::SelectCopySrcInput *input,
                                    Mapper::SelectCopySrcOutput *output,
                                    bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(COPY_SELECT_SOURCES_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_copy_sources(info, *op, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<CopyOp, Mapper::SelectCopySrcInput,
        Mapper::SelectCopySrcOutput, &MapperManager::invoke_select_copy_sources>
          continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_copy_speculate(CopyOp *op, 
                                              Mapper::SpeculativeOutput *output,
                                              bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(COPY_SPECULATE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->speculate(info, *op, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<CopyOp, Mapper::SpeculativeOutput,
                          &MapperManager::invoke_copy_speculate>
                            continuation(this, op, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_copy_report_profiling(CopyOp *op,
                                             Mapper::CopyProfilingInfo *input,
                                             bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(COPY_REPORT_PROFILING_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->report_profiling(info, *op, *input);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<CopyOp, Mapper::CopyProfilingInfo,
                          &MapperManager::invoke_copy_report_profiling>
                            continuation(this, op, input);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_close(CloseOp *op,
                                         Mapper::MapCloseInput *input,
                                         Mapper::MapCloseOutput *output,
                                         bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(MAP_CLOSE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->map_close(info, *op, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<CloseOp, Mapper::MapCloseInput, 
              Mapper::MapCloseOutput, &MapperManager::invoke_map_close>
                continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_close_sources(CloseOp *op,
                                         Mapper::SelectCloseSrcInput *input,
                                         Mapper::SelectCloseSrcOutput *output,
                                         bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(CLOSE_SELECT_SOURCES_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_close_sources(info, *op, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<CloseOp, Mapper::SelectCloseSrcInput,
        Mapper::SelectCloseSrcOutput, 
        &MapperManager::invoke_select_close_sources>
          continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_close_report_profiling(CloseOp *op,
                                          Mapper::CloseProfilingInfo *input,
                                          bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(CLOSE_REPORT_PROFILING_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->report_profiling(info, *op, *input);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<CloseOp, Mapper::CloseProfilingInfo,
                          &MapperManager::invoke_close_report_profiling>
                            continuation(this, op, input);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_acquire(AcquireOp *op,
                                           Mapper::MapAcquireInput *input,
                                           Mapper::MapAcquireOutput *output,
                                           bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(MAP_ACQUIRE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->map_acquire(info, *op, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<AcquireOp, Mapper::MapAcquireInput,
        Mapper::MapAcquireOutput, &MapperManager::invoke_map_acquire>
          continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_acquire_speculate(AcquireOp *op,
                                             Mapper::SpeculativeOutput *output,
                                             bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(ACQUIRE_SPECULATE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->speculate(info, *op, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<AcquireOp, Mapper::SpeculativeOutput,
                          &MapperManager::invoke_acquire_speculate>
                            continuation(this, op, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_acquire_report_profiling(AcquireOp *op,
                                         Mapper::AcquireProfilingInfo *input,
                                         bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(ACQUIRE_REPORT_PROFILING_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->report_profiling(info, *op, *input);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<AcquireOp, Mapper::AcquireProfilingInfo,
                          &MapperManager::invoke_acquire_report_profiling>
                            continuation(this, op, input);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_release(ReleaseOp *op,
                                           Mapper::MapReleaseInput *input,
                                           Mapper::MapReleaseOutput *output,
                                           bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(MAP_RELEASE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->map_release(info, *op, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<ReleaseOp, Mapper::MapReleaseInput,
        Mapper::MapReleaseOutput, &MapperManager::invoke_map_release>
          continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_release_sources(ReleaseOp *op,
                                       Mapper::SelectReleaseSrcInput *input,
                                       Mapper::SelectReleaseSrcOutput *output,
                                       bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(RELEASE_SELECT_SOURCES_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_release_sources(info, *op, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<ReleaseOp, Mapper::SelectReleaseSrcInput,
                          Mapper::SelectReleaseSrcOutput, 
                          &MapperManager::invoke_select_release_sources>
                            continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_release_speculate(ReleaseOp *op,
                                             Mapper::SpeculativeOutput *output,
                                             bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(RELEASE_SPECULATE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->speculate(info, *op, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<ReleaseOp, Mapper::SpeculativeOutput,
                          &MapperManager::invoke_release_speculate>
                            continuation(this, op, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_release_report_profiling(ReleaseOp *op,
                                         Mapper::ReleaseProfilingInfo *input,
                                         bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(RELEASE_REPORT_PROFILING_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->report_profiling(info, *op, *input);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<ReleaseOp, Mapper::ReleaseProfilingInfo,
                          &MapperManager::invoke_release_report_profiling>
                            continuation(this, op, input);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_configure_context(TaskOp *task,
                                         Mapper::ContextConfigOutput *output,
                                         bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(CONFIGURE_CONTEXT_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->configure_context(info, *task, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<TaskOp, Mapper::ContextConfigOutput,
                          &MapperManager::invoke_configure_context>
                            continuation(this, task, output);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_tunable_value(TaskOp *task,
                                     Mapper::SelectTunableInput *input,
                                     Mapper::SelectTunableOutput *output,
                                     bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(SELECT_TUNABLE_VALUE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_tunable_value(info, *task, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<TaskOp, Mapper::SelectTunableInput,
                          Mapper::SelectTunableOutput, 
                          &MapperManager::invoke_select_tunable_value>
                            continuation(this, task, input, output);
      continuation.defer(runtime, continuation_precondition, task);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_must_epoch(MustEpochOp *op,
                                            Mapper::MapMustEpochInput *input,
                                            Mapper::MapMustEpochOutput *output,
                                            bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(MAP_MUST_EPOCH_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->map_must_epoch(info, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation3<MustEpochOp, Mapper::MapMustEpochInput, 
                          Mapper::MapMustEpochOutput,
                          &MapperManager::invoke_map_must_epoch>
                            continuation(this, op, input, output);
      continuation.defer(runtime, continuation_precondition, op);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_dataflow_graph(
                                   Mapper::MapDataflowGraphInput *input,
                                   Mapper::MapDataflowGraphOutput *output,
                                   bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(MAP_DATAFLOW_GRAPH_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->map_dataflow_graph(info, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<Mapper::MapDataflowGraphInput, 
                          Mapper::MapDataflowGraphOutput,
                          &MapperManager::invoke_map_dataflow_graph>
                            continuation(this, input, output);
      continuation.defer(runtime, continuation_precondition);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_tasks_to_map(
                                    Mapper::SelectMappingInput *input,
                                    Mapper::SelectMappingOutput *output,
                                    bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(SELECT_TASKS_TO_MAP_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_tasks_to_map(info, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<Mapper::SelectMappingInput,
                          Mapper::SelectMappingOutput,
                          &MapperManager::invoke_select_tasks_to_map>
                            continuation(this, input, output);
      continuation.defer(runtime, continuation_precondition);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_steal_targets(
                                     Mapper::SelectStealingInput *input,
                                     Mapper::SelectStealingOutput *output,
                                     bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(SELECT_STEAL_TARGETS_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->select_steal_targets(info, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<Mapper::SelectStealingInput,
                          Mapper::SelectStealingOutput,
                          &MapperManager::invoke_select_steal_targets>
                            continuation(this, input, output);
      continuation.defer(runtime, continuation_precondition);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_permit_steal_request(
                                     Mapper::StealRequestInput *input,
                                     Mapper::StealRequestOutput *output,
                                     bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(PERMIT_STEAL_REQUEST_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->permit_steal_request(info, *input, *output);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation2<Mapper::StealRequestInput,
                          Mapper::StealRequestOutput,
                          &MapperManager::invoke_permit_steal_request>
                            continuation(this, input, output);
      continuation.defer(runtime, continuation_precondition);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_handle_message(Mapper::MapperMessage *message,
                                              bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(HANDLE_MESSAGE_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->handle_message(info, *message);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation1<Mapper::MapperMessage,
                          &MapperManager::invoke_handle_message>
                            continuation(this, message);
      continuation.defer(runtime, continuation_precondition);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_handle_task_result(
                                   Mapper::MapperTaskResult *result,
                                   bool first_invocation)
    //--------------------------------------------------------------------------
    {
      Event continuation_precondition = Event::NO_EVENT;
      MappingCallInfo *info = begin_mapper_call(HANDLE_TASK_RESULT_CALL,
                                first_invocation, continuation_precondition);
      if (info != NULL)
      {
        mapper->handle_task_result(info, *result);
        finish_mapper_call(info);
        return;
      }
#ifdef DEBUG_HIGH_LEVEL
      assert(first_invocation);
#endif
      MapperContinuation1<Mapper::MapperTaskResult,
                          &MapperManager::invoke_handle_task_result>
                            continuation(this, result);
      continuation.defer(runtime, continuation_precondition);
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
        constraints, regions, result, mapper_id, processor, acquire, priority);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl);
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
                      regions, result, mapper_id, processor, acquire, priority);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_or_create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    bool acquire, GCPriority priority)
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
                                      constraints, regions, result, created, 
                                      mapper_id, processor, acquire, priority);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_or_create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    bool acquire, GCPriority priority)
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
                                       layout_id, regions, result, created, 
                                       mapper_id, processor, acquire, priority);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_physical_instance(  
                                    MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire)
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
                                                     regions, result, acquire);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_physical_instance(  
                                    MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool acquire)
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
                                                     regions, result, acquire);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl);
      resume_mapper_call(ctx);
      return success;
    }

    //--------------------------------------------------------------------------
    void MapperManager::set_garbage_collection_priority(MappingCallInfo *ctx,
                           const MappingInstance &instance, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      PhysicalManager *manager = instance.impl;
      if (manager == NULL)
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
      if (manager == NULL)
        return true;
      // See if we already acquired it
      if (ctx->acquired_instances->find(manager) !=
          ctx->acquired_instances->end())
        return true;
      pause_mapper_call(ctx);
      if (manager->try_add_base_valid_ref(MAPPING_ACQUIRE_REF,
                                          !manager->is_owner()))
      {
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
        record_acquired_instance(ctx, instance.impl);
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
        std::map<PhysicalManager*,unsigned> &already_acquired = 
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
        std::map<PhysicalManager*,unsigned> &already_acquired = 
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
      std::map<PhysicalManager*,unsigned> &already_acquired = 
        *(info->acquired_instances);
      bool local_acquired = true;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        PhysicalManager *manager = instances[idx].impl;
        if (manager == NULL)
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
          already_acquired[manager] = 1;
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
            record_acquired_instance(info, *it); 
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
                                                 PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      if (manager == NULL)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx->acquired_instances != NULL);
#endif
      std::map<PhysicalManager*,unsigned> &acquired =*(ctx->acquired_instances);
      std::map<PhysicalManager*,unsigned>::iterator finder = 
        acquired.find(manager); 
      if (finder == acquired.end())
        acquired[manager] = 1; // first reference
      else
        finder->second++;
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_acquired_instance(MappingCallInfo *ctx,
                                                  PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      if (manager == NULL)
        return;
#ifdef DEBUG_HIGH_LEVEL
      assert(ctx->acquired_instances != NULL);
#endif
      std::map<PhysicalManager*,unsigned> &acquired =*(ctx->acquired_instances);
      std::map<PhysicalManager*,unsigned>::iterator finder = 
        acquired.find(manager);
      if (finder == acquired.end())
        return;
      // Release the refrences and then keep going, we know there is 
      // a resource reference so no need to check for deletion
      manager->remove_base_valid_ref(MAPPING_ACQUIRE_REF, finder->second);
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
#ifdef DEBUG_HIGH_LEVEL
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
    MappingCallInfo* MapperManager::allocate_call_info(MappingCallKind kind,
                                                       bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock m_lock(mapper_lock);
        return allocate_call_info(kind, false/*need lock*/);
      }
      if (!available_infos.empty())
      {
        MappingCallInfo *result = available_infos.back();
        available_infos.pop_back();
        result->kind = kind;
        return result;
      }
      return new MappingCallInfo(this, kind);
    }

    //--------------------------------------------------------------------------
    void MapperManager::free_call_info(MappingCallInfo *info, bool need_lock)
    //--------------------------------------------------------------------------
    {
      if (need_lock)
      {
        AutoLock m_lock(mapper_lock);
        free_call_info(info, false/*need lock*/);
      }
      info->resume = UserEvent::NO_USER_EVENT;
      available_infos.push_back(info);
    }

    //--------------------------------------------------------------------------
    /*static*/ const char* MapperManager::get_mapper_call_name(
                                                           MappingCallKind kind)
    //--------------------------------------------------------------------------
    {
      MAPPER_CALL_NAMES(call_names); 
#ifdef DEBUG_HIGH_LEVEL
      assert(kind < LAST_MAPPER_CALL);
#endif
      return call_names[kind];
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
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
      // No need to hold the lock here since we are exclusive
      return permit_reentrant;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::enable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
      // No need to hold the lock since we know we are exclusive 
      permit_reentrant = true;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::disable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
      // No need to hold the lock since we know we are exclusive
      if (permit_reentrant)
      {
        // If there are paused calls, we need to wait for them all 
        // to finish before we can continue execution 
        if (paused_calls > 0)
        {
#ifdef DEBUG_HIGH_LEVEL
          assert(!info->resume.exists());
#endif
          UserEvent ready_event = UserEvent::create_user_event();
          info->resume = ready_event;
          non_reentrant_calls.push_back(info);
          ready_event.wait();
          // When we wake up, we should be non-reentrant
#ifdef DEBUG_HIGH_LEVEL
          assert(!permit_reentrant);
#endif
        }
        else
          permit_reentrant = false;
      }
    }

    //--------------------------------------------------------------------------
    MappingCallInfo* SerializingManager::begin_mapper_call(MappingCallKind kind,
                                     bool first_invocation, Event &precondition)
    //--------------------------------------------------------------------------
    {
      Event wait_on = Event::NO_EVENT;  
      MappingCallInfo *result = NULL;
      // If this is the first invocation we have to ask for the lock
      // otherwise we know we already have it
      if (first_invocation)
      {
        precondition = mapper_lock.acquire(0, true/*exclusive*/);
        if (!precondition.has_triggered())
          return NULL;
      }
      result = allocate_call_info(kind, false/*need lock*/);
      // See if we are ready to run this or not
      if ((executing_call != NULL) || (!permit_reentrant && 
            ((paused_calls > 0) || !ready_calls.empty())))
      {
        // Put this on the list of pending calls
        result->resume = UserEvent::create_user_event();
        wait_on = result->resume;
        pending_calls.push_back(result);
      }
      else
        executing_call = result;
      mapper_lock.release();
      // If we have an event to wait on, then wait until we can execute
      if (wait_on.exists())
        wait_on.wait();
      return result;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::pause_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
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
#ifdef DEBUG_HIGH_LEVEL
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
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
    }

    //--------------------------------------------------------------------------
    void SerializingManager::finish_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_HIGH_LEVEL
      assert(executing_call == info);
#endif
      UserEvent to_trigger = UserEvent::NO_USER_EVENT;
      {
        AutoLock m_lock(mapper_lock);
        // See if can start a non-reentrant task
        if (!non_reentrant_calls.empty() && (paused_calls == 0) && 
            ready_calls.empty())
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
      }
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
          // TODO: error message for duplicate acquire
          assert(false);
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
          // Really bad if we can't find it in the set of current holders
          // TODO: put in an error message here
          assert(false);
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
                                     bool first_invocation, Event &precondition)
    //--------------------------------------------------------------------------
    {
      if (first_invocation)
      {
        precondition = mapper_lock.acquire(0, true/*exclusive*/);
        if (!precondition.has_triggered())
          return NULL;
      }
      MappingCallInfo *result = allocate_call_info(kind, false/*need lock*/);
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
    void ConcurrentManager::finish_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      std::vector<UserEvent> to_trigger;
      {
        AutoLock m_lock(mapper_lock);
        // Check to see if we need to release the lock for the mapper call
        std::set<MappingCallInfo*>::iterator finder = 
            current_holders.find(info);     
        if (finder != current_holders.end())
        {
          current_holders.erase(finder);
          release_lock(to_trigger);
        }
        free_call_info(info, false/*need lock*/);
      }
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
    MapperContinuation::MapperContinuation(MapperManager *man)
      : manager(man)
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
                          HLR_MAPPER_CONTINUATION_TASK_ID, op, precondition);
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
