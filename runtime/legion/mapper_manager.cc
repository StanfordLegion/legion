/* Copyright 2024 Stanford University, NVIDIA Corporation
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
#include "legion/legion_ops.h"
#include "legion/region_tree.h"
#include "legion/legion_tasks.h"
#include "legion/mapper_manager.h"
#include "legion/legion_instances.h"
#include "legion/garbage_collection.h"

namespace Legion {
  namespace Internal {

    LEGION_EXTERN_LOGGER_DECLARATIONS

    /////////////////////////////////////////////////////////////
    // Mapping Call Info 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MappingCallInfo::MappingCallInfo(MapperManager *man, MappingCallKind k,
                                     Operation *op, bool prioritize)
      : manager(man), resume(RtUserEvent::NO_RT_USER_EVENT), 
        kind(k), operation(op), acquired_instances((op == NULL) ? NULL :
            operation->get_acquired_instances_ref()), 
        start_time(0), reentrant_disabled(false)
    //--------------------------------------------------------------------------
    {
      manager->begin_mapper_call(this, prioritize);
    }

    //--------------------------------------------------------------------------
    MappingCallInfo::~MappingCallInfo(void)
    //--------------------------------------------------------------------------
    {
      manager->finish_mapper_call(this);
    }

    /////////////////////////////////////////////////////////////
    // Mapper Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapperManager::MapperManager(Runtime *rt, Mapping::Mapper *mp, 
                                 MapperID mid, Processor p, bool is_default)
      : runtime(rt), mapper(mp), mapper_id(mid), processor(p),
        profile_mapper(runtime->profiler != NULL),
        request_valid_instances(mp->request_valid_instances()),
        is_default_mapper(is_default)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(processor.exists());
#endif
      if (profile_mapper)
        runtime->profiler->record_mapper_name(mapper_id, processor, 
                                              get_mapper_name());
    }

    //--------------------------------------------------------------------------
    MapperManager::~MapperManager(void)
    //--------------------------------------------------------------------------
    {
      // We can now delete our mapper
      delete mapper;
    }

    //--------------------------------------------------------------------------
    const char* MapperManager::get_mapper_name(void)
    //--------------------------------------------------------------------------
    {
      return mapper->get_mapper_name();
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_task_options(TaskOp *task, 
                                  Mapper::TaskOptions &options, bool prioritize)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, SELECT_TASK_OPTIONS_CALL, task, prioritize);
      // If we have an info, we know we are good to go
      mapper->select_task_options(&ctx, *task, options);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_premap_task(TaskOp *task, 
                                           Mapper::PremapTaskInput &input,
                                           Mapper::PremapTaskOutput &output) 
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, PREMAP_TASK_CALL, task);
      mapper->premap_task(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_slice_task(TaskOp *task, 
                                          Mapper::SliceTaskInput &input,
                                          Mapper::SliceTaskOutput &output) 
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, SLICE_TASK_CALL, task);
      mapper->slice_task(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_task(TaskOp *task, 
                                        Mapper::MapTaskInput &input,
                                        Mapper::MapTaskOutput &output) 
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MAP_TASK_CALL, task);
      mapper->map_task(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_replicate_task(TaskOp *task,
                                     Mapper::ReplicateTaskInput &input,
                                     Mapper::ReplicateTaskOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, REPLICATE_TASK_CALL, task);
      mapper->replicate_task(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_task_variant(TaskOp *task,
                                            Mapper::SelectVariantInput &input,
                                            Mapper::SelectVariantOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, SELECT_VARIANT_CALL, task);
      mapper->select_task_variant(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_post_map_task(TaskOp *task, 
                                             Mapper::PostMapInput &input,
                                             Mapper::PostMapOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, POSTMAP_TASK_CALL, task);
      mapper->postmap_task(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_task_sources(TaskOp *task, 
                                    Mapper::SelectTaskSrcInput &input,
                                    Mapper::SelectTaskSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, TASK_SELECT_SOURCES_CALL, task);
      mapper->select_task_sources(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_task_sources(RemoteTaskOp *task, 
                                    Mapper::SelectTaskSrcInput &input,
                                    Mapper::SelectTaskSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, TASK_SELECT_SOURCES_CALL, task);
      mapper->select_task_sources(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_task_report_profiling(TaskOp *task, 
                                               Mapper::TaskProfilingInfo &input)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, TASK_REPORT_PROFILING_CALL, task);
      mapper->report_profiling(&ctx, *task, input);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_task_select_sharding_functor(TaskOp *task,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, TASK_SELECT_SHARDING_FUNCTOR_CALL, task);
      mapper->select_sharding_functor(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_inline(MapOp *op, 
                                          Mapper::MapInlineInput &input,
                                          Mapper::MapInlineOutput &output) 
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MAP_INLINE_CALL, op);
      mapper->map_inline(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_inline_sources(MapOp *op, 
                                      Mapper::SelectInlineSrcInput &input,
                                      Mapper::SelectInlineSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, INLINE_SELECT_SOURCES_CALL, op);
      mapper->select_inline_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_inline_sources(RemoteMapOp *op, 
                                      Mapper::SelectInlineSrcInput &input,
                                      Mapper::SelectInlineSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, INLINE_SELECT_SOURCES_CALL, op);
      mapper->select_inline_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_inline_report_profiling(MapOp *op, 
                                     Mapper::InlineProfilingInfo &input)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, INLINE_REPORT_PROFILING_CALL, op);
      mapper->report_profiling(&ctx, *op, input);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_copy(CopyOp *op,
                                        Mapper::MapCopyInput &input,
                                        Mapper::MapCopyOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MAP_COPY_CALL, op);
      mapper->map_copy(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_copy_sources(CopyOp *op,
                                    Mapper::SelectCopySrcInput &input,
                                    Mapper::SelectCopySrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, COPY_SELECT_SOURCES_CALL, op);
      mapper->select_copy_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_copy_sources(RemoteCopyOp *op,
                                    Mapper::SelectCopySrcInput &input,
                                    Mapper::SelectCopySrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, COPY_SELECT_SOURCES_CALL, op);
      mapper->select_copy_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_copy_report_profiling(CopyOp *op,
                                             Mapper::CopyProfilingInfo &input)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, COPY_REPORT_PROFILING_CALL, op);
      mapper->report_profiling(&ctx, *op, input);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_copy_select_sharding_functor(CopyOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, COPY_SELECT_SHARDING_FUNCTOR_CALL, op);
      mapper->select_sharding_functor(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_close_sources(CloseOp *op,
                                         Mapper::SelectCloseSrcInput &input,
                                         Mapper::SelectCloseSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, CLOSE_SELECT_SOURCES_CALL, op);
      mapper->select_close_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_close_sources(RemoteCloseOp *op,
                                         Mapper::SelectCloseSrcInput &input,
                                         Mapper::SelectCloseSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, CLOSE_SELECT_SOURCES_CALL, op);
      mapper->select_close_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_close_report_profiling(CloseOp *op,
                                          Mapper::CloseProfilingInfo &input)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, CLOSE_REPORT_PROFILING_CALL, op);
      mapper->report_profiling(&ctx, *op, input);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_close_select_sharding_functor(CloseOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, CLOSE_SELECT_SHARDING_FUNCTOR_CALL, op);
      mapper->select_sharding_functor(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_acquire(AcquireOp *op,
                                           Mapper::MapAcquireInput &input,
                                           Mapper::MapAcquireOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MAP_ACQUIRE_CALL, op);
      mapper->map_acquire(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_acquire_report_profiling(AcquireOp *op,
                                         Mapper::AcquireProfilingInfo &input)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, ACQUIRE_REPORT_PROFILING_CALL, op);
      mapper->report_profiling(&ctx, *op, input);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_acquire_select_sharding_functor(AcquireOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, ACQUIRE_SELECT_SHARDING_FUNCTOR_CALL, op);
      mapper->select_sharding_functor(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_release(ReleaseOp *op,
                                           Mapper::MapReleaseInput &input,
                                           Mapper::MapReleaseOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MAP_RELEASE_CALL, op);
      mapper->map_release(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_release_sources(ReleaseOp *op,
                                       Mapper::SelectReleaseSrcInput &input,
                                       Mapper::SelectReleaseSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, RELEASE_SELECT_SOURCES_CALL, op);
      mapper->select_release_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_release_sources(RemoteReleaseOp *op,
                                       Mapper::SelectReleaseSrcInput &input,
                                       Mapper::SelectReleaseSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, RELEASE_SELECT_SOURCES_CALL, op);
      mapper->select_release_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_release_report_profiling(ReleaseOp *op,
                                         Mapper::ReleaseProfilingInfo &input)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, RELEASE_REPORT_PROFILING_CALL, op);
      mapper->report_profiling(&ctx, *op, input);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_release_select_sharding_functor(ReleaseOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, RELEASE_SELECT_SHARDING_FUNCTOR_CALL, op);
      mapper->select_sharding_functor(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_partition_projection(
                          DependentPartitionOp *op,
                          Mapper::SelectPartitionProjectionInput &input,
                          Mapper::SelectPartitionProjectionOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, SELECT_PARTITION_PROJECTION_CALL, op);
      mapper->select_partition_projection(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_partition(DependentPartitionOp *op,
                                  Mapper::MapPartitionInput &input,
                                  Mapper::MapPartitionOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MAP_PARTITION_CALL, op);
      mapper->map_partition(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_partition_sources(
                                  DependentPartitionOp *op,
                                  Mapper::SelectPartitionSrcInput &input,
                                  Mapper::SelectPartitionSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, PARTITION_SELECT_SOURCES_CALL, op);
      mapper->select_partition_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_partition_sources(RemotePartitionOp *op,
                                  Mapper::SelectPartitionSrcInput &input,
                                  Mapper::SelectPartitionSrcOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, PARTITION_SELECT_SOURCES_CALL, op);
      mapper->select_partition_sources(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_partition_report_profiling(
                                         DependentPartitionOp *op,
                                         Mapper::PartitionProfilingInfo &input)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, PARTITION_REPORT_PROFILING_CALL, op);
      mapper->report_profiling(&ctx, *op, input);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_partition_select_sharding_functor(
                              DependentPartitionOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, PARTITION_SELECT_SHARDING_FUNCTOR_CALL, op);
      mapper->select_sharding_functor(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_fill_select_sharding_functor(FillOp *op,
                              Mapper::SelectShardingFunctorInput &input,
                              Mapper::SelectShardingFunctorOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, FILL_SELECT_SHARDING_FUNCTOR_CALL, op);
      mapper->select_sharding_functor(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_future_map_reduction(AllReduceOp *op,
                                       Mapper::FutureMapReductionInput &input,
                                       Mapper::FutureMapReductionOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MAP_FUTURE_MAP_REDUCTION_CALL, op);
      mapper->map_future_map_reduction(&ctx, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_configure_context(TaskOp *task,
                                         Mapper::ContextConfigOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, CONFIGURE_CONTEXT_CALL, task);
      mapper->configure_context(&ctx, *task, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_tunable_value(TaskOp *task,
                                     Mapper::SelectTunableInput &input,
                                     Mapper::SelectTunableOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, SELECT_TUNABLE_VALUE_CALL, task);
      mapper->select_tunable_value(&ctx, *task, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_must_epoch_select_sharding_functor(
                                MustEpochOp *op,
                                Mapper::SelectShardingFunctorInput &input,
                                Mapper::MustEpochShardingFunctorOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MUST_EPOCH_SELECT_SHARDING_FUNCTOR_CALL, op);
      mapper->select_sharding_functor(&ctx, *op, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_must_epoch(MustEpochOp *op,
                                            Mapper::MapMustEpochInput &input,
                                            Mapper::MapMustEpochOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MAP_MUST_EPOCH_CALL, op);
      mapper->map_must_epoch(&ctx, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_map_dataflow_graph(
                                   Mapper::MapDataflowGraphInput &input,
                                   Mapper::MapDataflowGraphOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MAP_DATAFLOW_GRAPH_CALL, NULL);
      mapper->map_dataflow_graph(&ctx, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_memoize_operation(Mappable *mappable,
                                                 Mapper::MemoizeInput &input,
                                                 Mapper::MemoizeOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, MEMOIZE_OPERATION_CALL, NULL);
      mapper->memoize_operation(&ctx, *mappable, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_tasks_to_map(
                                    Mapper::SelectMappingInput &input,
                                    Mapper::SelectMappingOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, SELECT_TASKS_TO_MAP_CALL, NULL);
      mapper->select_tasks_to_map(&ctx, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_select_steal_targets(
                                     Mapper::SelectStealingInput &input,
                                     Mapper::SelectStealingOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, SELECT_STEAL_TARGETS_CALL, NULL);
      mapper->select_steal_targets(&ctx, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_permit_steal_request(
                                     Mapper::StealRequestInput &input,
                                     Mapper::StealRequestOutput &output)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, PERMIT_STEAL_REQUEST_CALL, NULL);
      mapper->permit_steal_request(&ctx, input, output);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_handle_message(Mapper::MapperMessage *message,
                                              bool check_defer)
    //--------------------------------------------------------------------------
    {
      // Special case for handle message, always defer it if we are also
      // the sender in order to avoid deadlocks, same thing for any
      // local processor for non-reentrant mappers, have to use a test
      // for NULL pointer here since mapper continuation want
      // pointer arguments
      if (check_defer && 
          ((message->sender == processor) ||
           ((mapper->get_mapper_sync_model() == 
             Mapper::SERIALIZED_NON_REENTRANT_MAPPER_MODEL) && 
            runtime->is_local(message->sender))))
      {
        defer_message(message);
        return;
      }
      MappingCallInfo ctx(this, HANDLE_MESSAGE_CALL, NULL);
      mapper->handle_message(&ctx, *message);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_handle_task_result(
                                   Mapper::MapperTaskResult &result)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, HANDLE_TASK_RESULT_CALL, NULL);
      mapper->handle_task_result(&ctx, result);
    }

    //--------------------------------------------------------------------------
    void MapperManager::invoke_handle_instance_collection(
                                                      MappingInstance &instance)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo ctx(this, HANDLE_INSTANCE_COLLECTION_CALL, NULL);
      mapper->handle_instance_collection(&ctx, instance);
    }

    //--------------------------------------------------------------------------
    void MapperManager::notify_instance_deletion(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      // Get a reference in case we need to defer this
      MappingInstance instance(manager); 
      invoke_handle_instance_collection(instance);
    }

    //--------------------------------------------------------------------------
    void MapperManager::add_subscriber_reference(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      // Nothing to do currently
    }

    //--------------------------------------------------------------------------
    bool MapperManager::remove_subscriber_reference(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      // Nothing to do, make sure we don't get deleted
      return false;
    }

    //--------------------------------------------------------------------------
    void MapperManager::update_mappable_tag(MappingCallInfo *ctx,
                                 const Mappable &mappable, MappingTagID new_tag)
    //--------------------------------------------------------------------------
    {
      Mappable *map = const_cast<Mappable*>(&mappable);
      map->tag = new_tag;
    }

    //--------------------------------------------------------------------------
    void MapperManager::update_mappable_data(MappingCallInfo *ctx,
            const Mappable &mappable, const void *mapper_data, size_t data_size)
    //--------------------------------------------------------------------------
    {
      Mappable *map = const_cast<Mappable*>(&mappable);
      // Free the old buffer if there is one
      if (map->mapper_data != NULL)
        free(map->mapper_data);
      map->mapper_data_size = data_size;
      if (data_size > 0)
      {
        map->mapper_data = malloc(data_size);
        memcpy(map->mapper_data, mapper_data, data_size);
      }
      else
        map->mapper_data = NULL;
    }

    //--------------------------------------------------------------------------
    void MapperManager::send_message(MappingCallInfo *ctx, Processor target,
                const void *message, size_t message_size, unsigned message_kind)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->process_mapper_message(target, mapper_id, processor,
                                      message, message_size, message_kind);
      resume_mapper_call(ctx, MAPPER_SEND_MESSAGE_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::broadcast(MappingCallInfo *ctx, const void *message,
                          size_t message_size, unsigned message_kind, int radix)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->process_mapper_broadcast(mapper_id, processor, message,
                        message_size, message_kind, radix, 0/*index*/);
      resume_mapper_call(ctx, MAPPER_BROADCAST_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::pack_physical_instance(MappingCallInfo *ctx,
                                      Serializer &rez, MappingInstance instance)
    //--------------------------------------------------------------------------
    {
      // No need to even pause the mapper call here
      rez.serialize(instance.impl->did);
    }

    //--------------------------------------------------------------------------
    void MapperManager::unpack_physical_instance(MappingCallInfo *ctx,
                                 Deserializer &derez, MappingInstance &instance)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      DistributedID did;
      derez.deserialize(did);
      RtEvent ready;
      instance.impl = runtime->find_or_request_instance_manager(did, ready);
      if (ready.exists())
        ready.wait();
      resume_mapper_call(ctx, MAPPER_UNPACK_INSTANCE_CALL);
    }

    //--------------------------------------------------------------------------
    MapperEvent MapperManager::create_mapper_event(MappingCallInfo *ctx)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      MapperEvent result;
      result.impl = Runtime::create_rt_user_event();
      resume_mapper_call(ctx, MAPPER_CREATE_EVENT_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::has_mapper_event_triggered(MappingCallInfo *ctx,
                                                   MapperEvent event)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const bool triggered = event.impl.has_triggered();
      resume_mapper_call(ctx, MAPPER_HAS_TRIGGERED_CALL);
      return triggered;
    }
    
    //--------------------------------------------------------------------------
    void MapperManager::trigger_mapper_event(MappingCallInfo *ctx, 
                                             MapperEvent event)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      RtUserEvent to_trigger = event.impl;
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      resume_mapper_call(ctx, MAPPER_TRIGGER_EVENT_CALL);
    }
    
    //--------------------------------------------------------------------------
    void MapperManager::wait_on_mapper_event(MappingCallInfo *ctx,
                                             MapperEvent event)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      RtEvent wait_on = event.impl;
      if (wait_on.exists())
        wait_on.wait();
      resume_mapper_call(ctx, MAPPER_WAIT_EVENT_CALL);
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
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s requested execution "
                      "constraints for variant %d in mapper call %s, but "
                      "that variant does not exist.", mapper->get_mapper_name(),
                      vid, get_mapper_call_name(ctx->kind))
      const ExecutionConstraintSet &result = impl->get_execution_constraints();
      resume_mapper_call(ctx, MAPPER_FIND_EXECUTION_CONSTRAINTS_CALL);
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
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s requested task layout "
                      "constraints for variant %d in mapper call %s, but "
                      "that variant does not exist.", mapper->get_mapper_name(),
                      vid, get_mapper_call_name(ctx->kind))
      const TaskLayoutConstraintSet& result = impl->get_layout_constraints();
      resume_mapper_call(ctx, MAPPER_FIND_TASK_LAYOUT_CONSTRAINTS_CALL);
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
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s requested layout "
                      "constraints for layout ID %ld in mapper call %s, but "
                      "that layout constraint ID is invalid.",
                      mapper->get_mapper_name(), layout_id,
                      get_mapper_call_name(ctx->kind))
      resume_mapper_call(ctx, MAPPER_FIND_LAYOUT_CONSTRAINTS_CALL);
      return *constraints;
    }

    //--------------------------------------------------------------------------
    LayoutConstraintID MapperManager::register_layout(MappingCallInfo *ctx,
                                    const LayoutConstraintSet &constraints,
                                    FieldSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LayoutConstraints *cons = 
        runtime->register_layout(handle, constraints, false/*internal*/);
      resume_mapper_call(ctx, MAPPER_REGISTER_LAYOUT_CALL);
      return cons->layout_id;
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_layout(MappingCallInfo *ctx, 
                                       LayoutConstraintID layout_id)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->release_layout(layout_id);
      resume_mapper_call(ctx, MAPPER_RELEASE_LAYOUT_CALL);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::do_constraints_conflict(MappingCallInfo *ctx,
                               LayoutConstraintID set1, LayoutConstraintID set2,
                               const LayoutConstraint **conflict_constraint)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LayoutConstraints *c1 = 
        runtime->find_layout_constraints(set1, true/*can fail*/);
      LayoutConstraints *c2 = 
        runtime->find_layout_constraints(set2, true/*can fail*/);
      if ((c1 == NULL) || (c2 == NULL))
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s passed layout ID %ld "
                      "to conflict test in mapper call %s, but that layout ID "
                      "is invalid.", mapper->get_mapper_name(), 
                      (c1 == NULL) ? set1 : set2, 
                      get_mapper_call_name(ctx->kind))
      const bool result = 
        c1->conflicts(c2, 0/*dont care about dimensions*/, conflict_constraint);
      resume_mapper_call(ctx, MAPPER_CONSTRAINTS_CONFLICT_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::do_constraints_entail(MappingCallInfo *ctx,
                           LayoutConstraintID source, LayoutConstraintID target,
                           const LayoutConstraint **failed_constraint)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LayoutConstraints *c1 = 
        runtime->find_layout_constraints(source, true/*can fail*/);
      LayoutConstraints *c2 = 
        runtime->find_layout_constraints(target, true/*can fail*/);
      if ((c1 == NULL) || (c2 == NULL))
        REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                      "Invalid mapper request: mapper %s passed layout ID %ld "
                      "to entailment test in mapper call %s, but that layout "
                      "ID is invalid.", mapper->get_mapper_name(), 
                      (c1 == NULL) ? source : target, 
                      get_mapper_call_name(ctx->kind))
      const bool result = 
        c1->entails(c2, 0/*don't care about dimensions*/, failed_constraint);
      resume_mapper_call(ctx, MAPPER_CONSTRAINTS_ENTAIL_CALL);
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
      resume_mapper_call(ctx, MAPPER_FIND_VALID_VARIANTS_CALL);
    }
    
    //--------------------------------------------------------------------------
    const char* MapperManager::find_task_variant_name(
                     MappingCallInfo *ctx, TaskID task_id, VariantID variant_id)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantImpl *impl = runtime->find_variant_impl(task_id, variant_id);
      const char *name = impl->get_name();
      resume_mapper_call(ctx, MAPPER_FIND_TASK_VARIANT_NAME_CALL);
      return name;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::is_leaf_variant(MappingCallInfo *ctx,
                                        TaskID task_id, VariantID variant_id)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantImpl *impl = runtime->find_variant_impl(task_id, variant_id);
      bool result = impl->is_leaf();
      resume_mapper_call(ctx, MAPPER_IS_LEAF_VARIANT_CALL);
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
      resume_mapper_call(ctx, MAPPER_IS_INNER_VARIANT_CALL);
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
      resume_mapper_call(ctx, MAPPER_IS_IDEMPOTENT_VARIANT_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::is_replicable_variant(MappingCallInfo *ctx,
                                           TaskID task_id, VariantID variant_id)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantImpl *impl = runtime->find_variant_impl(task_id, variant_id);
      bool result = impl->is_replicable();
      resume_mapper_call(ctx, MAPPER_IS_REPLICABLE_VARIANT_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    VariantID MapperManager::register_task_variant(MappingCallInfo *ctx,
                                  const TaskVariantRegistrar &registrar,
                                  const CodeDescriptor &realm_desc,
                                  const void *user_data, size_t user_len,
                                  size_t return_type_size, bool has_return_type)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      VariantID result = runtime->register_variant(registrar, user_data,
                user_len, realm_desc, return_type_size, has_return_type);
      resume_mapper_call(ctx, MAPPER_REGISTER_TASK_VARIANT_CALL);
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
            InstanceManager *manager = instances[idx].impl;
            if (manager->conflicts(constraints,  NULL))
            {
              conflicts = true;
              break;
            }
            if (!constraints->specialized_constraint.is_virtual() &&
                (constraints->specialized_constraint.is_exact() ||
                 constraints->padding_constraint.delta.get_dim() > 0))
            {
              std::vector<LogicalRegion> regions_to_check(1,
                        task.regions[lay_it->first].region);
              PhysicalManager *phy = manager->as_physical_manager();
              if (!phy->meets_regions(regions_to_check,
                    constraints->specialized_constraint.is_exact(),
                    &constraints->padding_constraint.delta))
              {
                conflicts = true;
                break;
              }
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
      resume_mapper_call(ctx, MAPPER_FILTER_VARIANTS_CALL);
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
            InstanceManager *manager = it->impl;
            if (manager->conflicts(constraints, NULL))
              it = instances.erase(it);
            else if (!constraints->specialized_constraint.is_virtual() &&
                      (constraints->specialized_constraint.is_exact() ||
                       constraints->padding_constraint.delta.get_dim() > 0))
            {
              std::vector<LogicalRegion> regions_to_check(1,
                        task.regions[lay_it->first].region);
              PhysicalManager *phy = manager->as_physical_manager();
              if (!phy->meets_regions(regions_to_check,
                    constraints->specialized_constraint.is_exact(),
                    &constraints->padding_constraint.delta))
                it = instances.erase(it);
              else
                it++;
            }
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
          InstanceManager *manager = it->impl;
          manager->remove_space_fields(missing);
          if (missing.empty())
            break;
        }
      }
      resume_mapper_call(ctx, MAPPER_FILTER_INSTANCES_CALL);
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
          InstanceManager *manager = it->impl;
          if (manager->conflicts(constraints, NULL))
            it = instances.erase(it);
          else if (!constraints->specialized_constraint.is_virtual() &&
                    (constraints->specialized_constraint.is_exact() ||
                     constraints->padding_constraint.delta.get_dim() > 0))
          {
            std::vector<LogicalRegion> regions_to_check(1,
                      task.regions[lay_it->first].region);
            PhysicalManager *phy = manager->as_physical_manager();
            if (!phy->meets_regions(regions_to_check,
                  constraints->specialized_constraint.is_exact(),
                  &constraints->padding_constraint.delta))
              it = instances.erase(it);
            else
              it++;
          }
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
        InstanceManager *manager = it->impl;
        manager->remove_space_fields(missing_fields);
        if (missing_fields.empty())
          break;
      }
      resume_mapper_call(ctx, MAPPER_FILTER_INSTANCES_CALL);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds, size_t *footprint,
                                    const LayoutConstraint **unsat)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      check_region_consistency(ctx, "create_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to create_physical_instance "
                        "in unsupported mapper call %s in mapper %s", 
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      bool success = runtime->create_physical_instance(target_memory, 
        constraints, regions, result, processor, acquire, priority, 
        tight_region_bounds, unsat, footprint, (ctx->operation == NULL) ? 
          0 : ctx->operation->get_unique_op_id());
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, true/*created*/);
      resume_mapper_call(ctx, MAPPER_CREATE_PHYSICAL_INSTANCE_CALL);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result,
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds, size_t *footprint,
                                    const LayoutConstraint **unsat)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      check_region_consistency(ctx, "create_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to create_physical_instance "
                        "in unsupported mapper call %s in mapper %s", 
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      LayoutConstraints *cons = runtime->find_layout_constraints(layout_id);
      bool success = runtime->create_physical_instance(target_memory, cons,
                      regions, result, processor, acquire, priority,
                      tight_region_bounds, unsat, footprint,
                      (ctx->operation == NULL) ? 0 :
                        ctx->operation->get_unique_op_id());
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, true/*created*/);
      resume_mapper_call(ctx, MAPPER_CREATE_PHYSICAL_INSTANCE_CALL);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_or_create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints, 
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds, size_t *footprint,
                                    const LayoutConstraint **unsat)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      check_region_consistency(ctx, "find_or_create_physical_instance",
                               regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_or_create_physical"
                        "_instance in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      bool success = runtime->find_or_create_physical_instance(target_memory,
                constraints, regions, result, created, processor, 
                acquire, priority, tight_region_bounds, unsat, footprint,
                (ctx->operation == NULL) ? 0 :
                 ctx->operation->get_unique_op_id());
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, created);
      resume_mapper_call(ctx, MAPPER_FIND_OR_CREATE_PHYSICAL_INSTANCE_CALL);
      return success;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::find_or_create_physical_instance(
                                    MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    MappingInstance &result, bool &created, 
                                    bool acquire, GCPriority priority,
                                    bool tight_region_bounds, size_t *footprint,
                                    const LayoutConstraint **unsat)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return false;
      if (regions.empty())
        return false;
      check_region_consistency(ctx, "find_or_create_physical_instance",
                               regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_or_create_physical"
                        "_instance in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      LayoutConstraints *cons = runtime->find_layout_constraints(layout_id);
      bool success = runtime->find_or_create_physical_instance(target_memory,
                 cons, regions, result, created, processor,
                 acquire, priority, tight_region_bounds, unsat, footprint,
                 (ctx->operation == NULL) ? 0 : 
                  ctx->operation->get_unique_op_id());
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, created);
      resume_mapper_call(ctx, MAPPER_FIND_OR_CREATE_PHYSICAL_INSTANCE_CALL);
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
      check_region_consistency(ctx, "find_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_physical_instance "
                        "in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      bool success = runtime->find_physical_instance(target_memory, constraints,
                                 regions, result, acquire, tight_region_bounds);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, false/*created*/);
      resume_mapper_call(ctx, MAPPER_FIND_PHYSICAL_INSTANCE_CALL);
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
      check_region_consistency(ctx, "find_physical_instance", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_physical_instance "
                        "in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      LayoutConstraints *cons = runtime->find_layout_constraints(layout_id);
      bool success = runtime->find_physical_instance(target_memory, cons,
                          regions, result, acquire, tight_region_bounds);
      if (success && acquire)
        record_acquired_instance(ctx, result.impl, false/*created*/);
      resume_mapper_call(ctx, MAPPER_FIND_PHYSICAL_INSTANCE_CALL);
      return success;
    }

    //--------------------------------------------------------------------------
    void MapperManager::find_physical_instances(  
                                    MappingCallInfo *ctx, Memory target_memory,
                                    const LayoutConstraintSet &constraints,
                                    const std::vector<LogicalRegion> &regions,
                                    std::vector<MappingInstance> &results, 
                                    bool acquire, bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return;
      check_region_consistency(ctx, "find_physical_instances", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_physical_instances "
                        "in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      const size_t initial_size = results.size();
      runtime->find_physical_instances(target_memory, constraints, regions, 
                                    results, acquire, tight_region_bounds);
      if ((initial_size < results.size()) && acquire)
      {
        for (unsigned idx = initial_size; idx < results.size(); idx++)
          record_acquired_instance(ctx, results[idx].impl, false/*created*/);
      }
      resume_mapper_call(ctx, MAPPER_FIND_PHYSICAL_INSTANCES_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::find_physical_instances(  
                                    MappingCallInfo *ctx, Memory target_memory,
                                    LayoutConstraintID layout_id,
                                    const std::vector<LogicalRegion> &regions,
                                    std::vector<MappingInstance> &results, 
                                    bool acquire, bool tight_region_bounds)
    //--------------------------------------------------------------------------
    {
      if (!target_memory.exists())
        return;
      check_region_consistency(ctx, "find_physical_instances", regions);
      if (acquire && (ctx->acquired_instances == NULL))
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request to find_physical_instances "
                        "in unsupported mapper call %s in mapper %s",
                        get_mapper_call_name(ctx->kind), get_mapper_name());
        acquire = false;
      }
      pause_mapper_call(ctx);
      LayoutConstraints *cons = runtime->find_layout_constraints(layout_id);
      const size_t initial_size = results.size();
      runtime->find_physical_instances(target_memory, cons, regions, 
                              results, acquire, tight_region_bounds);
      if ((initial_size < results.size()) && acquire)
      {
        for (unsigned idx = initial_size; idx < results.size(); idx++)
          record_acquired_instance(ctx, results[idx].impl, false/*created*/);
      }
      resume_mapper_call(ctx, MAPPER_FIND_PHYSICAL_INSTANCES_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::set_garbage_collection_priority(MappingCallInfo *ctx,
                           const MappingInstance &instance, GCPriority priority)
    //--------------------------------------------------------------------------
    {
      InstanceManager *man = instance.impl;
      if (man->is_virtual_manager())
        return;
      pause_mapper_call(ctx);
      PhysicalManager *manager = man->as_physical_manager();
      // Ignore garbage collection priorities on external instances
      if (!manager->is_external_instance())
      {
        const RtEvent ready = manager->set_garbage_collection_priority(
                mapper_id, processor, runtime->address_space, priority);
        if (ready.exists() && !ready.has_triggered())
          ready.wait();
      }
      else
        REPORT_LEGION_WARNING(LEGION_WARNING_EXTERNAL_GARBAGE_PRIORITY,
            "Ignoring request for mapper %s to set garbage collection "
            "priority on an external instance", get_mapper_name())
      resume_mapper_call(ctx, MAPPER_SET_GC_PRIORITY_CALL);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_instance(MappingCallInfo *ctx,
                                         const MappingInstance &instance)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      InstanceManager *man = instance.impl;
      // virtual instances are easy
      if (man->is_virtual_manager())
        return true;
      PhysicalManager *manager = man->as_physical_manager();
      // See if we already acquired it
      if (ctx->acquired_instances->find(manager) !=
          ctx->acquired_instances->end())
        return true;
      pause_mapper_call(ctx);
      if (manager->acquire_instance(MAPPING_ACQUIRE_REF))
      {
        record_acquired_instance(ctx, manager, false/*created*/);
        resume_mapper_call(ctx, MAPPER_ACQUIRE_INSTANCE_CALL);
        return true;
      }
      else
      {
        resume_mapper_call(ctx, MAPPER_ACQUIRE_INSTANCE_CALL);
        return false;
      }
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_instances(MappingCallInfo *ctx,
                                  const std::vector<MappingInstance> &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      // Quick fast path
      if (instances.size() == 1)
        return acquire_instance(ctx, instances[0]);
      pause_mapper_call(ctx);
      const bool all_acquired = perform_acquires(ctx, instances);
      resume_mapper_call(ctx, MAPPER_ACQUIRE_INSTANCES_CALL);
      return all_acquired;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_and_filter_instances(MappingCallInfo *ctx,
                                        std::vector<MappingInstance> &instances,
                                        const bool filter_acquired_instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      // Quick fast path
      if (instances.size() == 1)
      {
        bool result = acquire_instance(ctx, instances[0]);
        if (result)
        {
          if (filter_acquired_instances)
            instances.clear();
        }
        else
        {
          if (!filter_acquired_instances)
            instances.clear();
        }
        return result;
      }
      pause_mapper_call(ctx);
      // Figure out which instances we need to acquire and sort by memories
      std::vector<unsigned> to_erase;
      const bool all_acquired =
        perform_acquires(ctx, instances, &to_erase, filter_acquired_instances);
      // Filter any invalid local instances
      if (!to_erase.empty())
      {
        // Erase from the back
        for (std::vector<unsigned>::const_reverse_iterator it =
              to_erase.rbegin(); it != to_erase.rend(); it++)
          instances.erase(instances.begin()+(*it));
        to_erase.clear();
      }
      resume_mapper_call(ctx, MAPPER_ACQUIRE_AND_FILTER_INSTANCES_CALL);
      return all_acquired;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_instances(MappingCallInfo *ctx,
                    const std::vector<std::vector<MappingInstance> > &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      pause_mapper_call(ctx); 
      // Figure out which instances we need to acquire and sort by memories
      bool all_acquired = true;
      for (std::vector<std::vector<MappingInstance> >::const_iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        if (!perform_acquires(ctx, *it))
          all_acquired = false;
      }
      resume_mapper_call(ctx, MAPPER_ACQUIRE_INSTANCES_CALL);
      return all_acquired;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_and_filter_instances(MappingCallInfo *ctx,
                          std::vector<std::vector<MappingInstance> > &instances,
                          const bool filter_acquired_instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      pause_mapper_call(ctx);
      // Figure out which instances we need to acquire and sort by memories
      bool all_acquired = true;
      std::vector<unsigned> to_erase;
      for (std::vector<std::vector<MappingInstance> >::iterator it = 
            instances.begin(); it != instances.end(); it++)
      {
        if (!perform_acquires(ctx, *it, &to_erase, filter_acquired_instances))
        {
          all_acquired = false;
          // Erase from the back
          for (std::vector<unsigned>::const_reverse_iterator rit = 
                to_erase.rbegin(); rit != to_erase.rend(); rit++)
            it->erase(it->begin()+(*rit));
          to_erase.clear();
        }
      }
      resume_mapper_call(ctx, MAPPER_ACQUIRE_AND_FILTER_INSTANCES_CALL);
      return all_acquired;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::perform_acquires(MappingCallInfo *info,
                                  const std::vector<MappingInstance> &instances,
                                  std::vector<unsigned> *to_erase,
                                  const bool filter_acquired_instances)
    //--------------------------------------------------------------------------
    {
      std::map<PhysicalManager*,unsigned> &already_acquired =
        *(info->acquired_instances);
      bool all_acquired = true;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        const MappingInstance &inst = instances[idx];
        if (!inst.exists())
          continue;
        InstanceManager *man = inst.impl;
        if (man->is_virtual_manager())
          continue;
        PhysicalManager *manager = man->as_physical_manager();
        if (already_acquired.find(manager) != already_acquired.end())
        {
          if ((to_erase != NULL) && filter_acquired_instances)
            to_erase->push_back(idx);
          continue;
        }
        // Try to add an acquired reference immediately
        // If we're remote it has to be valid already to be sound, but if
        // we're local whatever works
        if (manager->acquire_instance(MAPPING_ACQUIRE_REF))
        {
          // We already know it wasn't there before
          already_acquired[manager] = 1;
          if ((to_erase != NULL) && filter_acquired_instances)
            to_erase->push_back(idx);
        }
        else
        {
          all_acquired = false;
          if ((to_erase != NULL) && !filter_acquired_instances)
            to_erase->push_back(idx);
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
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_RELEASE_REQUEST,
                        "Ignoring release request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return;
      }
      pause_mapper_call(ctx);
      release_acquired_instance(ctx, instance.impl); 
      resume_mapper_call(ctx, MAPPER_RELEASE_INSTANCE_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_instances(MappingCallInfo *ctx,
                                 const std::vector<MappingInstance> &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_RELEASE_REQUEST,
                        "Ignoring release request in unsupported mapper call "
                        "%s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return;
      }
      pause_mapper_call(ctx);
      for (unsigned idx = 0; idx < instances.size(); idx++)
        release_acquired_instance(ctx, instances[idx].impl);
      resume_mapper_call(ctx, MAPPER_RELEASE_INSTANCES_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_instances(MappingCallInfo *ctx, 
                   const std::vector<std::vector<MappingInstance> > &instances)
    //--------------------------------------------------------------------------
    {
      if (ctx->acquired_instances == NULL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_RELEASE_REQUEST,
                        "Ignoring release request in unsupported mapper call "
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
      resume_mapper_call(ctx, MAPPER_RELEASE_INSTANCES_CALL);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::subscribe(MappingCallInfo *ctx,
                                  const MappingInstance &instance)
    //--------------------------------------------------------------------------
    {
      if ((instance.impl == NULL) || instance.impl->is_virtual_manager())
        return false;
      pause_mapper_call(ctx);
      PhysicalManager *manager = instance.impl->as_physical_manager();
      const bool result = manager->register_deletion_subscriber(this);
      resume_mapper_call(ctx, MAPPER_SUBSCRIBE_INSTANCE_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::unsubscribe(MappingCallInfo *ctx,
                                    const MappingInstance &instance)
    //--------------------------------------------------------------------------
    {
      if ((instance.impl == NULL) || instance.impl->is_virtual_manager())
        return;
      pause_mapper_call(ctx);
      PhysicalManager *manager = instance.impl->as_physical_manager();
      manager->unregister_deletion_subscriber(this);
      resume_mapper_call(ctx, MAPPER_UNSUBSCRIBE_INSTANCE_CALL);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::collect_instance(MappingCallInfo *ctx, 
                                         const MappingInstance &instance)
    //--------------------------------------------------------------------------
    {
      if ((instance.impl == NULL) || instance.impl->is_virtual_manager() ||
          instance.impl->is_external_instance())
        return false;
      pause_mapper_call(ctx);
      PhysicalManager *manager = instance.impl->as_physical_manager();
      RtEvent collected;
      const bool result = manager->collect(collected);
      if (result)
      {
        // Tell the memory that the instance has been collected
        std::vector<PhysicalManager*> collected_instance(1, manager);
        manager->memory_manager->notify_collected_instances(collected_instance);
        // Wait for the collection to be done 
        collected.wait();
      }
      resume_mapper_call(ctx, MAPPER_COLLECT_INSTANCE_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::collect_instances(MappingCallInfo *ctx,
                                  const std::vector<MappingInstance> &instances,
                                  std::vector<bool> &collected)
    //--------------------------------------------------------------------------
    {
      collected.resize(instances.size(), false);
      if (instances.empty())
        return;
      pause_mapper_call(ctx);
      std::vector<RtEvent> wait_for;
      std::map<MemoryManager*,std::vector<PhysicalManager*> > to_notify;
      for (unsigned idx = 0; idx < instances.size(); idx++)
      {
        collected[idx] = false;
        InstanceManager *inst = instances[idx].impl;
        if ((inst == NULL) || inst->is_virtual_manager() || 
            inst->is_external_instance())
          continue;
        RtEvent instance_collected;
        PhysicalManager *manager = inst->as_physical_manager();
        if (manager->collect(instance_collected))
        {
          collected[idx] = true;
          to_notify[manager->memory_manager].push_back(manager);
          if (instance_collected.exists())
            wait_for.push_back(instance_collected);
        }
      }
      // Notify all the memory managers of the collection
      for (std::map<MemoryManager*,
                    std::vector<PhysicalManager*> >::const_iterator it =
            to_notify.begin(); it != to_notify.end(); it++)
        it->first->notify_collected_instances(it->second);
      if (!wait_for.empty())
      {
        const RtEvent wait_on = Runtime::merge_events(wait_for);
        wait_on.wait();
      }
      resume_mapper_call(ctx, MAPPER_COLLECT_INSTANCES_CALL);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_future(MappingCallInfo *ctx,
                                       const Future &future, Memory memory)
    //--------------------------------------------------------------------------
    {
      if ((future.impl == NULL) || !memory.exists())
        return false;
      if (ctx->kind != MAP_TASK_CALL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire future request in unsupported mapper "
                        "call %s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      pause_mapper_call(ctx);
      const bool result = future.impl->find_or_create_application_instance(
                                memory, ctx->operation->get_unique_op_id()); 
      resume_mapper_call(ctx, MAPPER_ACQUIRE_FUTURE_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::acquire_pool(MappingCallInfo *ctx,
                                     Memory memory, const PoolBounds &bounds)
    //--------------------------------------------------------------------------
    {
      if (!memory.exists() || (bounds.size == 0))
        return false;
      // Only support this in map-task calls
      if (ctx->kind != MAP_TASK_CALL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring acquire pool request in unsupported mapper "
                        "call %s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return false;
      }
      pause_mapper_call(ctx);
#ifdef DEBUG_LEGION
      assert(ctx->operation != NULL);
      SingleTask *task = dynamic_cast<SingleTask*>(ctx->operation);
      assert(task != NULL);
#else
      SingleTask *task = static_cast<SingleTask*>(ctx->operation);
#endif
      const bool result = task->acquire_leaf_memory_pool(memory, bounds);
      resume_mapper_call(ctx, MAPPER_ACQUIRE_POOL_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_pool(MappingCallInfo *ctx, Memory memory)
    //--------------------------------------------------------------------------
    {
      if (!memory.exists())
        return;
      // Only support this in map-task calls
      if (ctx->kind != MAP_TASK_CALL)
      {
        REPORT_LEGION_WARNING(LEGION_WARNING_IGNORING_ACQUIRE_REQUEST,
                        "Ignoring release pool request in unsupported mapper "
                        "call %s in mapper %s", get_mapper_call_name(ctx->kind),
                        get_mapper_name());
        return;
      }
      pause_mapper_call(ctx);
#ifdef DEBUG_LEGION
      assert(ctx->operation != NULL);
      SingleTask *task = dynamic_cast<SingleTask*>(ctx->operation);
      assert(task != NULL);
#else
      SingleTask *task = static_cast<SingleTask*>(ctx->operation);
#endif
      task->release_leaf_memory_pool(memory);
      resume_mapper_call(ctx, MAPPER_RELEASE_POOL_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::record_acquired_instance(MappingCallInfo *ctx,
                                             InstanceManager *man, bool created)
    //--------------------------------------------------------------------------
    {
      if (man->is_virtual_manager())
        return;
      PhysicalManager *manager = man->as_physical_manager();
#ifdef DEBUG_LEGION
      assert(ctx->acquired_instances != NULL);
#endif
      std::map<PhysicalManager*,unsigned> &acquired =
        *(ctx->acquired_instances);
      std::map<PhysicalManager*,unsigned>::iterator finder = 
        acquired.find(manager);
      if (finder == acquired.end())
        acquired[manager] = 1;
      else
        finder->second++;
    }

    //--------------------------------------------------------------------------
    void MapperManager::release_acquired_instance(MappingCallInfo *ctx,
                                                  InstanceManager *man)
    //--------------------------------------------------------------------------
    {
      if (man->is_virtual_manager())
        return;
      PhysicalManager *manager = man->as_physical_manager();
#ifdef DEBUG_LEGION
      assert(ctx->acquired_instances != NULL);
#endif
      std::map<PhysicalManager*,unsigned> &acquired = 
        *(ctx->acquired_instances);
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
        if (!regions[idx].exists())
          continue;
        if (tree_id > 0)
        {
          RegionTreeID other_id = regions[idx].get_tree_id();
          if (other_id != tree_id)
            REPORT_LEGION_ERROR(ERROR_INVALID_ARGUMENTS_TO_MAPPER_RUNTIME,
                          "Invalid region arguments passed to %s in "
                          "mapper call %s of mapper %s. All region arguments "
                          "must be from the same region tree (%d != %d).",
                          call_name, get_mapper_call_name(info->kind),
                          mapper->get_mapper_name(), tree_id, other_id)
        }
        else
          tree_id = regions[idx].get_tree_id();
      }
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::create_index_space(MappingCallInfo *ctx,
                                                 const Domain &domain,
                                                 TypeTag type_tag,
                                                 const char *prov)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Provenance *provenance = NULL;
      if (prov != NULL)
        provenance = runtime->find_or_create_provenance(prov, strlen(prov));
      const IndexSpace result(runtime->get_unique_index_space_id(),
                    runtime->get_unique_index_tree_id(), type_tag);
      const DistributedID did = runtime->get_available_distributed_id();
      runtime->forest->create_index_space(result, &domain, did, provenance);
      if ((provenance != NULL) && provenance->remove_reference())
        delete provenance;
      resume_mapper_call(ctx, MAPPER_CREATE_INDEX_SPACE_CALL);
      return result; 
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::union_index_spaces(MappingCallInfo *ctx,
                 const std::vector<IndexSpace> &sources, const char *provenance)
    //--------------------------------------------------------------------------
    {
      if (sources.empty())
        return IndexSpace::NO_SPACE;
      pause_mapper_call(ctx);
      bool none_exists = true;
      for (std::vector<IndexSpace>::const_iterator it = 
            sources.begin(); it != sources.end(); it++)
      {
        if (none_exists && it->exists())
          none_exists = false;
        if (sources[0].get_type_tag() != it->get_type_tag())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'union_index_spaces' "
                        "performed in mapper %s", get_mapper_name())
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      const IndexSpace result(runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), sources[0].get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
      AutoProvenance prov(provenance);
      runtime->forest->create_union_space(result, did, prov, sources);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(result.get_id(),
                    runtime->address_space, provenance);
      resume_mapper_call(ctx, MAPPER_UNION_INDEX_SPACES_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::intersect_index_spaces(MappingCallInfo *ctx,
                 const std::vector<IndexSpace> &sources, const char *provenance)
    //--------------------------------------------------------------------------
    {
      if (sources.empty())
        return IndexSpace::NO_SPACE;
      pause_mapper_call(ctx);
      bool none_exists = true;
      for (std::vector<IndexSpace>::const_iterator it = 
            sources.begin(); it != sources.end(); it++)
      {
        if (none_exists && it->exists())
          none_exists = false;
        if (sources[0].get_type_tag() != it->get_type_tag())
          REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'intersect_index_spaces' "
                        "performed in mapper %s", get_mapper_name())
      }
      if (none_exists)
        return IndexSpace::NO_SPACE;
      const IndexSpace result(runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), sources[0].get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
      AutoProvenance prov(provenance);
      runtime->forest->create_intersection_space(result, did, prov, sources);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(result.get_id(),
                    runtime->address_space, provenance);
      resume_mapper_call(ctx, MAPPER_INTERSECT_INDEX_SPACES_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::subtract_index_spaces(MappingCallInfo *ctx,
                      IndexSpace left, IndexSpace right, const char *provenance)
    //--------------------------------------------------------------------------
    {
      if (!left.exists())
        return IndexSpace::NO_SPACE;
      pause_mapper_call(ctx);
      if (right.exists() && left.get_type_tag() != right.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'create_difference_spaces' "
                        "performed in mapper %s", get_mapper_name())
      const IndexSpace result(runtime->get_unique_index_space_id(),
          runtime->get_unique_index_tree_id(), left.get_type_tag());
      const DistributedID did = runtime->get_available_distributed_id();
      AutoProvenance prov(provenance);
      runtime->forest->create_difference_space(result, did, prov,
                                               left, right);
      if (runtime->legion_spy_enabled)
        LegionSpy::log_top_index_space(result.get_id(),
                    runtime->address_space, provenance);
      resume_mapper_call(ctx, MAPPER_SUBTRACT_INDEX_SPACES_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::is_index_space_empty(MappingCallInfo *ctx,
                                             IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      if (!handle.exists())
        return true;
      pause_mapper_call(ctx);
      IndexSpaceNode *node = runtime->forest->get_node(handle);
      bool result = node->is_empty();
      resume_mapper_call(ctx, MAPPER_INDEX_SPACE_EMPTY_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::index_spaces_overlap(MappingCallInfo *ctx,
                                             IndexSpace one, IndexSpace two)
    //--------------------------------------------------------------------------
    {
      if (!one.exists() || !two.exists())
        return false;
      pause_mapper_call(ctx);
      if (one.get_type_tag() != two.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'index_spaces_overlap' "
                        "performed in mapper %s", get_mapper_name())
      IndexSpaceNode *n1 = runtime->forest->get_node(one);
      IndexSpaceNode *n2 = runtime->forest->get_node(two);
      IndexSpaceExpression *overlap = 
        runtime->forest->intersect_index_spaces(n1, n2);
      const bool result = !overlap->is_empty();
      resume_mapper_call(ctx, MAPPER_INDEX_SPACES_OVERLAP_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::index_space_dominates(MappingCallInfo *ctx,
                                              IndexSpace left, IndexSpace right)
    //--------------------------------------------------------------------------
    {
      if (!left.exists())
        return true;
      if (!right.exists())
        return false;
      pause_mapper_call(ctx);
      if (left.get_type_tag() != right.get_type_tag())
        REPORT_LEGION_ERROR(ERROR_DYNAMIC_TYPE_MISMATCH,
                        "Dynamic type mismatch in 'index_spaces_dominates' "
                        "performed in mapper %s", get_mapper_name())
      IndexSpaceNode *n1 = runtime->forest->get_node(left);
      IndexSpaceNode *n2 = runtime->forest->get_node(right);
      IndexSpaceExpression *difference =
        runtime->forest->subtract_index_spaces(n1, n2);
      const bool result = difference->is_empty();
      resume_mapper_call(ctx, MAPPER_INDEX_SPACE_DOMINATES_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::has_index_partition(MappingCallInfo *ctx,
                                            IndexSpace parent, Color color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool result = runtime->has_index_partition(parent, color);
      resume_mapper_call(ctx, MAPPER_HAS_INDEX_PARTITION_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperManager::get_index_partition(MappingCallInfo *ctx,
                                                      IndexSpace parent, 
                                                      Color color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexPartition result = runtime->get_index_partition(parent, color);
      resume_mapper_call(ctx, MAPPER_GET_INDEX_PARTITION_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_index_subspace(MappingCallInfo *ctx,
                                                 IndexPartition p, Color c)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Point<1,coord_t> color(c);
      IndexSpace result = runtime->get_index_subspace(p, &color,
                    NT_TemplateHelper::encode_tag<1,coord_t>());
      resume_mapper_call(ctx, MAPPER_GET_INDEX_SUBSPACE_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_index_subspace(MappingCallInfo *ctx,
                                                 IndexPartition p, 
                                                 const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexSpace result = IndexSpace::NO_SPACE;
      switch (color.get_dim())
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            Point<DIM,coord_t> point(color); \
            result = runtime->get_index_subspace(p, &point, \
                NT_TemplateHelper::encode_tag<DIM,coord_t>()); \
            break; \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      resume_mapper_call(ctx, MAPPER_GET_INDEX_SUBSPACE_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::has_multiple_domains(MappingCallInfo *ctx,
                                             IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      // Never have multiple domains
      return false;
    }

    //--------------------------------------------------------------------------
    Domain MapperManager::get_index_space_domain(MappingCallInfo *ctx,
                                                 IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Domain result = Domain::NO_DOMAIN;
      switch (NT_TemplateHelper::get_dim(handle.get_type_tag()))
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            DomainT<DIM,coord_t> realm_is; \
            const TypeTag tag = NT_TemplateHelper::encode_tag<DIM,coord_t>(); \
            runtime->get_index_space_domain(handle, &realm_is, tag); \
            result = realm_is; \
            break; \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      resume_mapper_call(ctx, MAPPER_GET_INDEX_SPACE_DOMAIN_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::get_index_space_domains(MappingCallInfo *ctx,
                                                IndexSpace handle,
                                                std::vector<Domain> &domains)
    //--------------------------------------------------------------------------
    {
      domains.push_back(get_index_space_domain(ctx, handle));
    }

    //--------------------------------------------------------------------------
    Domain MapperManager::get_index_partition_color_space(MappingCallInfo *ctx,
                                                          IndexPartition p)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Domain result = runtime->get_index_partition_color_space(p);
      resume_mapper_call(ctx, MAPPER_GET_INDEX_PARTITION_CS_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_index_partition_color_space_name(
                                         MappingCallInfo *ctx, IndexPartition p)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexSpace result = runtime->get_index_partition_color_space_name(p);
      resume_mapper_call(ctx, MAPPER_GET_INDEX_PARTITION_CS_NAME_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::get_index_space_partition_colors(MappingCallInfo *ctx,
                                     IndexSpace handle, std::set<Color> &colors)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->get_index_space_partition_colors(handle, colors);
      resume_mapper_call(ctx, MAPPER_GET_INDEX_SPACE_PARTITION_COLORS_CALL);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::is_index_partition_disjoint(MappingCallInfo *ctx,
                                                    IndexPartition p)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool result = runtime->is_index_partition_disjoint(p);
      resume_mapper_call(ctx, MAPPER_IS_INDEX_PARTITION_DISJOINT_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::is_index_partition_complete(MappingCallInfo *ctx,
                                                    IndexPartition p)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool result = runtime->is_index_partition_complete(p);
      resume_mapper_call(ctx, MAPPER_IS_INDEX_PARTITION_COMPLETE_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_index_space_color(MappingCallInfo *ctx,
                                               IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Point<1,coord_t> point;
      runtime->get_index_space_color_point(handle, &point,
                NT_TemplateHelper::encode_tag<1,coord_t>());
      resume_mapper_call(ctx, MAPPER_GET_INDEX_SPACE_COLOR_CALL);
      return point[0];
    }

    //--------------------------------------------------------------------------
    DomainPoint MapperManager::get_index_space_color_point(MappingCallInfo *ctx,
                                                           IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      DomainPoint result = runtime->get_index_space_color_point(handle);
      resume_mapper_call(ctx, MAPPER_GET_INDEX_SPACE_COLOR_POINT_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_index_partition_color(MappingCallInfo *ctx,
                                                   IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Color result = runtime->get_index_partition_color(handle);
      resume_mapper_call(ctx, MAPPER_GET_INDEX_PARTITION_COLOR_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexSpace MapperManager::get_parent_index_space(MappingCallInfo *ctx,
                                                     IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexSpace result = runtime->get_parent_index_space(handle);
      resume_mapper_call(ctx, MAPPER_GET_PARENT_INDEX_SPACE_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::has_parent_index_partition(MappingCallInfo *ctx,
                                                   IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool result = runtime->has_parent_index_partition(handle);
      resume_mapper_call(ctx, MAPPER_HAS_PARENT_INDEX_PARTITION_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    IndexPartition MapperManager::get_parent_index_partition(
                                        MappingCallInfo *ctx, IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      IndexPartition result = runtime->get_parent_index_partition(handle);
      resume_mapper_call(ctx, MAPPER_GET_PARENT_INDEX_PARTITION_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned MapperManager::get_index_space_depth(MappingCallInfo *ctx,
                                                  IndexSpace handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      unsigned result = runtime->get_index_space_depth(handle);
      resume_mapper_call(ctx, MAPPER_GET_INDEX_SPACE_DEPTH_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    unsigned MapperManager::get_index_partition_depth(MappingCallInfo *ctx,
                                                      IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      unsigned result = runtime->get_index_partition_depth(handle);
      resume_mapper_call(ctx, MAPPER_GET_INDEX_PARTITION_DEPTH_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    size_t MapperManager::get_field_size(MappingCallInfo *ctx,
                                         FieldSpace handle, FieldID fid)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      size_t result = runtime->get_field_size(handle, fid);
      resume_mapper_call(ctx, MAPPER_GET_FIELD_SIZE_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    void MapperManager::get_field_space_fields(MappingCallInfo *ctx,
                                FieldSpace handle, std::vector<FieldID> &fields)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      runtime->get_field_space_fields(handle, fields);
      resume_mapper_call(ctx, MAPPER_GET_FIELD_SPACE_FIELDS_CALL);
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_logical_partition(MappingCallInfo *ctx,
                                                          LogicalRegion parent,
                                                          IndexPartition handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalPartition result = runtime->get_logical_partition(parent, handle);
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_PARTITION_CALL);
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
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_PARTITION_BY_COLOR_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_logical_partition_by_color(
              MappingCallInfo *ctx, LogicalRegion par, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
#ifdef DEBUG_LEGION
      assert((color.get_dim() == 0) || (color.get_dim() == 1));
#endif
      LogicalPartition result = 
        runtime->get_logical_partition_by_color(par, color[0]);
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_PARTITION_BY_COLOR_CALL);
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
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_PARTITION_BY_TREE_CALL);
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
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_SUBREGION_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_logical_subregion_by_color(
                        MappingCallInfo *ctx, LogicalPartition par, Color color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Point<1,coord_t> point(color);
      LogicalRegion result = runtime->get_logical_subregion_by_color(par,
                      &point, NT_TemplateHelper::encode_tag<1,coord_t>());
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_SUBREGION_BY_COLOR_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_logical_subregion_by_color(
           MappingCallInfo *ctx, LogicalPartition par, const DomainPoint &color)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalRegion result = LogicalRegion::NO_REGION;
      switch (color.get_dim())
      {
#define DIMFUNC(DIM) \
        case DIM: \
          { \
            Point<DIM,coord_t> point(color); \
            result = runtime->get_logical_subregion_by_color(par, &point, \
                              NT_TemplateHelper::encode_tag<DIM,coord_t>()); \
            break; \
          }
        LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
        default:
          assert(false);
      }
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_SUBREGION_BY_COLOR_CALL);
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
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_SUBREGION_BY_TREE_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_logical_region_color(MappingCallInfo *ctx,
                                                  LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Point<1,coord_t> point;
      runtime->get_logical_region_color(handle, &point, 
            NT_TemplateHelper::encode_tag<1,coord_t>());
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_REGION_COLOR_CALL);
      return point[0];
    }

    //--------------------------------------------------------------------------
    DomainPoint MapperManager::get_logical_region_color_point(
                                                           MappingCallInfo *ctx,
                                                           LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      DomainPoint result = runtime->get_logical_region_color_point(handle);
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_REGION_COLOR_POINT_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    Color MapperManager::get_logical_partition_color(MappingCallInfo *ctx,
                                                     LogicalPartition handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      Color result = runtime->get_logical_partition_color(handle);
      resume_mapper_call(ctx, MAPPER_GET_LOGICAL_PARTITION_COLOR_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalRegion MapperManager::get_parent_logical_region(MappingCallInfo *ctx,
                                                          LogicalPartition part)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalRegion result = runtime->get_parent_logical_region(part);
      resume_mapper_call(ctx, MAPPER_GET_PARENT_LOGICAL_REGION_CALL);
      return result;
    }
    
    //--------------------------------------------------------------------------
    bool MapperManager::has_parent_logical_partition(MappingCallInfo *ctx,
                                                     LogicalRegion handle)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool result = runtime->has_parent_logical_partition(handle);
      resume_mapper_call(ctx, MAPPER_HAS_PARENT_LOGICAL_PARTITION_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    LogicalPartition MapperManager::get_parent_logical_partition(
                                          MappingCallInfo *ctx, LogicalRegion r)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      LogicalPartition result = runtime->get_parent_logical_partition(r);
      resume_mapper_call(ctx, MAPPER_GET_PARENT_LOGICAL_PARTITION_CALL);
      return result;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        TaskID task_id, SemanticTag tag, const void *&result, size_t &size,
        bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool ok = runtime->retrieve_semantic_information(task_id, tag,
					     result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx, MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return ok;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        IndexSpace handle, SemanticTag tag, const void *&result, size_t &size,
        bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool ok = runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx, MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return ok;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        IndexPartition handle, SemanticTag tag, const void *&result, 
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool ok = runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx, MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return ok;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        FieldSpace handle, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool ok = runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx, MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return ok;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        FieldSpace handle, FieldID fid, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool ok = runtime->retrieve_semantic_information(handle, fid,
					     tag, result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx, MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return ok;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        LogicalRegion handle, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool ok = runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx, MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return ok;
    }

    //--------------------------------------------------------------------------
    bool MapperManager::retrieve_semantic_information(MappingCallInfo *ctx,
        LogicalPartition handle, SemanticTag tag, const void *&result,
        size_t &size, bool can_fail, bool wait_until_ready)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      bool ok = runtime->retrieve_semantic_information(handle, tag,
					     result, size,
                                             can_fail, wait_until_ready);
      resume_mapper_call(ctx, MAPPER_RETRIEVE_SEMANTIC_INFO_CALL);
      return ok;
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, TaskID task_id,
                                      const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(task_id, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
      resume_mapper_call(ctx, MAPPER_RETRIEVE_NAME_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, IndexSpace handle,
                                      const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
      resume_mapper_call(ctx, MAPPER_RETRIEVE_NAME_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, 
                                      IndexPartition handle,const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
      resume_mapper_call(ctx, MAPPER_RETRIEVE_NAME_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, FieldSpace handle,
                                      const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
      resume_mapper_call(ctx, MAPPER_RETRIEVE_NAME_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, FieldSpace handle,
                                      FieldID fid, const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, fid, 
          LEGION_NAME_SEMANTIC_TAG, name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
      resume_mapper_call(ctx, MAPPER_RETRIEVE_NAME_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx, 
                                      LogicalRegion handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
      resume_mapper_call(ctx, MAPPER_RETRIEVE_NAME_CALL);
    }

    //--------------------------------------------------------------------------
    void MapperManager::retrieve_name(MappingCallInfo *ctx,
                                   LogicalPartition handle, const char *&result)
    //--------------------------------------------------------------------------
    {
      pause_mapper_call(ctx);
      const void *name; size_t dummy_size;
      runtime->retrieve_semantic_information(handle, LEGION_NAME_SEMANTIC_TAG,
                                             name, dummy_size, false, false);
      static_assert(sizeof(result) == sizeof(name));
      memcpy(&result, &name, sizeof(result));
      resume_mapper_call(ctx, MAPPER_RETRIEVE_NAME_CALL);
    }

    //--------------------------------------------------------------------------
    bool MapperManager::is_MPI_interop_configured(void)
    //--------------------------------------------------------------------------
    {
      return runtime->is_MPI_interop_configured();
    }

    //--------------------------------------------------------------------------
    const std::map<int,AddressSpace>& MapperManager::find_forward_MPI_mapping(
                                                           MappingCallInfo *ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->find_forward_MPI_mapping();
    }

    //--------------------------------------------------------------------------
    const std::map<AddressSpace,int>& MapperManager::find_reverse_MPI_mapping(
                                                           MappingCallInfo *ctx)
    //--------------------------------------------------------------------------
    {
      return runtime->find_reverse_MPI_mapping();
    }

    //--------------------------------------------------------------------------
    int MapperManager::find_local_MPI_rank(void)
    //--------------------------------------------------------------------------
    {
      return runtime->find_local_MPI_rank();
    }

    //--------------------------------------------------------------------------
    /*static*/ const char* MapperManager::get_mapper_call_name(
                                                           MappingCallKind kind)
    //--------------------------------------------------------------------------
    {
      static MAPPER_CALL_NAMES(call_names); 
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
      DeferMessageArgs args(this, message->sender, message->kind, 
                            malloc(message->size), message->size,
                            message->broadcast);
      memcpy(args.message, message->message, args.size);
      runtime->issue_runtime_meta_task(args, LG_RESOURCE_PRIORITY);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MapperManager::handle_deferred_message(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferMessageArgs *margs = (const DeferMessageArgs*)args;
      Mapper::MapperMessage message;
      message.sender = margs->sender;
      message.kind = margs->kind;
      message.message = margs->message;
      message.size = margs->size;
      message.broadcast = margs->broadcast;
      margs->manager->invoke_handle_message(&message, false/*no check*/);
      // Then free up the allocated memory
      free(margs->message);
    }

    //--------------------------------------------------------------------------
    void MapperManager::process_advertisement(Processor advertiser)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
#ifdef DEBUG_LEGION
      assert(steal_blacklist.find(advertiser) !=
             steal_blacklist.end());
#endif
      steal_blacklist.erase(advertiser);
    }

    //--------------------------------------------------------------------------
    void MapperManager::perform_stealing(
                                     std::multimap<Processor,MapperID> &targets)
    //--------------------------------------------------------------------------
    {
      Mapper::SelectStealingInput steal_input;
      Mapper::SelectStealingOutput steal_output;
      {
        AutoLock m_lock(mapper_lock, 1, false/*exclusive*/);
        steal_input.blacklist = steal_blacklist; 
      }
      invoke_select_steal_targets(steal_input, steal_output);
      if (steal_output.targets.empty())
        return;
      // Retake the lock and process the results
      AutoLock m_lock(mapper_lock);
      for (std::set<Processor>::const_iterator it = 
            steal_output.targets.begin(); it != 
            steal_output.targets.end(); it++)
      {
        if (it->exists() && ((*it) != processor) &&
            (steal_blacklist.find(*it) == steal_blacklist.end()))
        {
          targets.insert(std::pair<Processor,MapperID>(*it,mapper_id));
          steal_blacklist.insert(*it);
        }
      }
    }

    //--------------------------------------------------------------------------
    void MapperManager::process_failed_steal(Processor thief)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
#ifdef DEBUG_LEGION
      assert(failed_thiefs.find(thief) == failed_thiefs.end());
#endif
      failed_thiefs.insert(thief);
    }

    //--------------------------------------------------------------------------
    void MapperManager::perform_advertisements(
                                            std::set<Processor> &failed_waiters)
    //--------------------------------------------------------------------------
    {
      AutoLock m_lock(mapper_lock);
      failed_waiters = failed_thiefs;
      failed_thiefs.clear();
    }

    /////////////////////////////////////////////////////////////
    // Serializing Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    SerializingManager::SerializingManager(Runtime *rt, Mapping::Mapper *mp,
             MapperID map_id, Processor p, bool init_reentrant, bool def)
      : MapperManager(rt,mp,map_id,p,def),executing_call(NULL),paused_calls(0),
        allow_reentrant(init_reentrant), permit_reentrant(init_reentrant)
    //--------------------------------------------------------------------------
    {
      pending_pause_call.store(false);
      pending_finish_call.store(false);
    }

    //--------------------------------------------------------------------------
    SerializingManager::SerializingManager(const SerializingManager &rhs)
      : MapperManager(NULL, NULL, 0, Processor::NO_PROC, false), 
        allow_reentrant(false)
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
      REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                          "Illegal 'lock_mapper' call performed in mapper %s "
                          "with the serialized synchronization model. Use the "
                          "'disable_reentrant' call instead.",get_mapper_name())
    }

    //--------------------------------------------------------------------------
    void SerializingManager::unlock_mapper(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                          "Illegal 'unlock_mapper' call performed in mapper %s "
                          "with the serialized synchronization model. Use the "
                          "'enable_reentrant' call instead.", get_mapper_name())
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
      if (!allow_reentrant)
        REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                        "Illegal 'enable_reentrant' call performed in mapper "
                        "%s with the SERIALIZED_NON_REENTRANT_MAPPER_MODEL. "
                        "Reentrant calls are never allowed with this model.", 
                        get_mapper_name())
      else if (!info->reentrant_disabled)
        REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                        "Illegal 'disable_reentrant' call performed in mapper "
                        "%s. Reentrant calls were already enabled and we do "
                        "not support nested calls to enable them.",
                        get_mapper_name())
      info->reentrant_disabled = false;
      AutoLock m_lock(mapper_lock);
      permit_reentrant = true;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::disable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(executing_call == info);
#endif
      if (!allow_reentrant)
        REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                        "Illegal 'disable_reentrant' call performed in mapper "
                        "%s with the SERIALIZED_NON_REENTRANT_MAPPER_MODEL. "
                        "Reentrant calls are already disallowed with this "
                        "model.", get_mapper_name())
      else if (info->reentrant_disabled)
        REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                        "Illegal 'disable_reentrant' call performed in mapper "
                        "%s. Reentrant calls were already disabled and we do "
                        "not support nested calls to disable them.",
                        get_mapper_name())
      info->reentrant_disabled = true;
      AutoLock m_lock(mapper_lock);
      permit_reentrant = false;
    }

    //--------------------------------------------------------------------------
    void SerializingManager::begin_mapper_call(MappingCallInfo *info,
                                               bool prioritize)
    //--------------------------------------------------------------------------
    {
      RtEvent precondition;
      RtUserEvent to_trigger;
      {
        AutoLock m_lock(mapper_lock);
        // See if there is a pending call for us to handle
        if (pending_pause_call.load())
          to_trigger = complete_pending_pause_mapper_call();
        else if (pending_finish_call.load())
          to_trigger = complete_pending_finish_mapper_call();
        // See if we are ready to run this or not
        if ((executing_call != NULL) || (!permit_reentrant && 
              ((paused_calls > 0) || !ready_calls.empty())))
        {
          // Put this on the list of pending calls
          info->resume = Runtime::create_rt_user_event();
          precondition = info->resume;
          if (prioritize)
            pending_calls.push_front(info);
          else
            pending_calls.push_back(info);
        }
        else
          executing_call = info;
      }
      // Wake up a pending mapper call to run if necessary
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
      if (profile_mapper)
      {
        if (is_default_mapper)
          runtime->profiler->issue_default_mapper_warning(info->operation,
                                      get_mapper_call_name(info->kind));
        info->start_time = Realm::Clock::current_time_in_nanoseconds();
      }
      if (precondition.exists() && !precondition.has_triggered())
        precondition.wait();
#ifdef DEBUG_LEGION
      assert(executing_call == info);
#endif
    }

    //--------------------------------------------------------------------------
    void SerializingManager::pause_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (profile_mapper)
        info->pause_time = Realm::Clock::current_time_in_nanoseconds();
      if (executing_call != info)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_CONTENT,
                      "Invalid mapper context passed to mapper_rt "
                      "call by mapper %s. Mapper contexts are only valid "
                      "for the mapper call to which they are passed. They "
                      "cannot be stored beyond the lifetime of the "
                      "mapper call.", mapper->get_mapper_name())
#ifdef DEBUG_LEGION
      assert(!pending_pause_call.load());
#endif
      // Set the flag indicating there is a paused mapper call that
      // needs to be handled, do this asynchronoulsy and check to 
      // see if we lost the race later
      pending_pause_call.store(true); 
      // We definitely know we can't start any non_reentrant calls
      // Screw fairness, we care about throughput, see if there are any
      // pending calls to wake up, and then go to sleep ourself
      RtUserEvent to_trigger;
      {
        AutoLock m_lock(mapper_lock);
        // See if we lost the race
        if (pending_pause_call.load())
          to_trigger = complete_pending_pause_mapper_call(); 
      }
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    void SerializingManager::resume_mapper_call(MappingCallInfo *info,
                                                RuntimeCallKind kind)
    //--------------------------------------------------------------------------
    {
      if (profile_mapper)
        runtime->profiler->record_runtime_call(kind, info->pause_time,
            Realm::Clock::current_time_in_nanoseconds());
      // See if we are ready to be woken up
      RtEvent wait_on;
      {
        AutoLock m_lock(mapper_lock);
#ifdef DEBUG_LEGION
        assert(paused_calls > 0);
#endif
        paused_calls--;
        // If the executing call is ourself then we are the only ones
        // that are allowed to resume because reentrant is disabled
        if (executing_call != info)
        {
          if (executing_call != NULL)
          {
            info->resume = Runtime::create_rt_user_event();
            wait_on = info->resume;
            ready_calls.push_back(info);
          }
          else
            executing_call = info;
        }
      }
      if (wait_on.exists())
        wait_on.wait();
#ifdef DEBUG_LEGION
      assert(executing_call == info);
#endif
    }

    //--------------------------------------------------------------------------
    void SerializingManager::finish_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(executing_call == info);
#endif
      // Record our finish time when we're done
      if (profile_mapper)
        runtime->profiler->record_mapper_call(mapper_id, processor, info->kind,
            (info->operation == NULL) ? 0 : info->operation->get_unique_op_id(),
            info->start_time, Realm::Clock::current_time_in_nanoseconds());
      // Set this flag asynchronously without the lock, there will
      // be a race to see who gets the lock next and therefore can
      // do the rest of the finish mapper call routine, we do this
      // to avoid the priority inversion that can occur where this
      // lock acquire gets stuck behind a bunch of pending ones
#ifdef DEBUG_LEGION
      assert(!pending_finish_call.load());
#endif
      pending_finish_call.store(true);
      RtUserEvent to_trigger;
      {
        AutoLock m_lock(mapper_lock);
        // We've got the lock, see if we won the race to the flag
        if (pending_finish_call.load())
          to_trigger = complete_pending_finish_mapper_call();  
      }
      // Wake up the next task if necessary
      if (to_trigger.exists())
        Runtime::trigger_event(to_trigger);
    }

    //--------------------------------------------------------------------------
    RtUserEvent SerializingManager::complete_pending_pause_mapper_call(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(pending_pause_call.load());
      assert(!pending_finish_call.load());
#endif
      pending_pause_call.store(false);
      // Increment the count of the paused mapper calls
      paused_calls++;
      if (permit_reentrant)
      {
        if (!pending_calls.empty())
        {
          executing_call = pending_calls.front();
          pending_calls.pop_front();
          return executing_call->resume;
        }
        else if (!ready_calls.empty())
        {
          executing_call = ready_calls.front();
          ready_calls.pop_front();
          return executing_call->resume;
        }
        // If we are allowing reentrant calls then clear the executing
        // call which will allow other resuming calls to run
        executing_call = NULL;
      }
      // No one to wake up
      return RtUserEvent::NO_RT_USER_EVENT;
    }

    //--------------------------------------------------------------------------
    RtUserEvent SerializingManager::complete_pending_finish_mapper_call(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(!pending_pause_call.load());
      assert(pending_finish_call.load());
      assert(executing_call != NULL);
#endif
      pending_finish_call.store(false);
      // If we allow reentrant calls then reset whether we are permitting
      // reentrant calls in case the user forgot to do it at the end of call
      if (allow_reentrant && executing_call->reentrant_disabled)
      {
#ifdef DEBUG_LEGION
        assert(!permit_reentrant);
#endif
        permit_reentrant = true;
      }
      if (!pending_calls.empty())
      {
        executing_call = pending_calls.front();
        pending_calls.pop_front();
        return executing_call->resume;
      }
      else if (!ready_calls.empty())
      {
        executing_call = ready_calls.front();
        ready_calls.pop_front();
        return executing_call->resume;
      }
      else
      {
        executing_call = NULL;
        return RtUserEvent::NO_RT_USER_EVENT;
      }
    }

    /////////////////////////////////////////////////////////////
    // Concurrent Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    ConcurrentManager::ConcurrentManager(Runtime *rt, Mapping::Mapper *mp,
                                         MapperID map_id, Processor p, bool def)
      : MapperManager(rt, mp, map_id, p, def), lock_state(UNLOCKED_STATE)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ConcurrentManager::ConcurrentManager(const ConcurrentManager &rhs)
      : MapperManager(NULL, NULL, 0, Processor::NO_PROC, false)
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
      RtEvent wait_on;
      {
        AutoLock m_lock(mapper_lock); 
        if (current_holders.find(info) != current_holders.end())
          REPORT_LEGION_ERROR(ERROR_INVALID_DUPLICATE_MAPPER,
                        "Invalid duplicate mapper lock request in mapper call "
                        "%s for mapper %s", get_mapper_call_name(info->kind),
                        mapper->get_mapper_name())
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
                info->resume = Runtime::create_rt_user_event();
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
              info->resume = Runtime::create_rt_user_event();
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
      std::vector<RtUserEvent> to_trigger;
      {
        AutoLock m_lock(mapper_lock);
        std::set<MappingCallInfo*>::iterator finder = 
          current_holders.find(info);
        if (finder == current_holders.end())
          REPORT_LEGION_ERROR(ERROR_INVALID_UNLOCK_MAPPER,
                        "Invalid unlock mapper call with no prior lock call "
                        "in mapper call %s for mapper %s",
                        get_mapper_call_name(info->kind),
                        mapper->get_mapper_name())
        current_holders.erase(finder);
        // See if we can now give the lock to someone else
        if (current_holders.empty())
          release_lock(to_trigger);
      }
      if (!to_trigger.empty())
      {
        for (std::vector<RtUserEvent>::const_iterator it = 
              to_trigger.begin(); it != to_trigger.end(); it++)
          Runtime::trigger_event(*it);
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
      REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                          "Illegal 'enable_reentrant' call performed in mapper "
                          "%s with the concurrent synchronization model. Use "
                          "the 'unlock_mapper' call instead.",get_mapper_name())
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::disable_reentrant(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                          "Illegal 'disable_reentrant' call performed in mapper"
                          " %s with the concurrent synchronization model. Use "
                          "the 'lock_mapper' call instead.", get_mapper_name())
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::begin_mapper_call(MappingCallInfo *info,
                                              bool prioritize)
    //--------------------------------------------------------------------------
    {
      // Record our mapper start time when we're ready to run
      if (profile_mapper)
      {
        if (is_default_mapper)
          runtime->profiler->issue_default_mapper_warning(info->operation,
                                        get_mapper_call_name(info->kind));
        info->start_time = Realm::Clock::current_time_in_nanoseconds();
      }
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::pause_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      if (profile_mapper)
        info->pause_time = Realm::Clock::current_time_in_nanoseconds();
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::resume_mapper_call(MappingCallInfo *info,
                                               RuntimeCallKind kind)
    //--------------------------------------------------------------------------
    {
      if (profile_mapper)
        runtime->profiler->record_runtime_call(kind, info->pause_time,
            Realm::Clock::current_time_in_nanoseconds());
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::finish_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Record our finish time when we are done
      if (profile_mapper)
        runtime->profiler->record_mapper_call(mapper_id, processor, info->kind,
            (info->operation == NULL) ? 0 : info->operation->get_unique_op_id(),
            info->start_time, Realm::Clock::current_time_in_nanoseconds());
      std::vector<RtUserEvent> to_trigger;
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
      }
      if (!to_trigger.empty())
      {
        for (std::vector<RtUserEvent>::const_iterator it = 
              to_trigger.begin(); it != to_trigger.end(); it++)
          Runtime::trigger_event(*it);
      }
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::release_lock(std::vector<RtUserEvent> &to_trigger)
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

  };
}; // namespace Legion
