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
        start_time(0), reentrant(manager->initially_reentrant)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(implicit_mapper_call == NULL);
#endif
      implicit_mapper_call = this;
      manager->begin_mapper_call(this, prioritize);
    }

    //--------------------------------------------------------------------------
    MappingCallInfo::~MappingCallInfo(void)
    //--------------------------------------------------------------------------
    {
#ifdef DEBUG_LEGION
      assert(implicit_mapper_call == this);
#endif
      manager->finish_mapper_call(this);
      implicit_mapper_call = NULL;
    }

    //--------------------------------------------------------------------------
    void MappingCallInfo::record_acquired_instance(InstanceManager *man,
                                                   bool created)
    //--------------------------------------------------------------------------
    {
      if (man->is_virtual_manager())
        return;
      PhysicalManager *manager = man->as_physical_manager();
#ifdef DEBUG_LEGION
      assert(acquired_instances != NULL);
#endif
      std::map<PhysicalManager*,unsigned> &acquired =
        *(acquired_instances);
      std::map<PhysicalManager*,unsigned>::iterator finder = 
        acquired.find(manager);
      if (finder == acquired.end())
        acquired[manager] = 1;
      else
        finder->second++;
    }

    //--------------------------------------------------------------------------
    void MappingCallInfo::release_acquired_instance(InstanceManager *man)
    //--------------------------------------------------------------------------
    {
      if (man->is_virtual_manager())
        return;
      PhysicalManager *manager = man->as_physical_manager();
#ifdef DEBUG_LEGION
      assert(acquired_instances != NULL);
#endif
      std::map<PhysicalManager*,unsigned> &acquired = 
        *(acquired_instances);
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
    bool MappingCallInfo::perform_acquires(
                                const std::vector<MappingInstance> &instances,
                                std::vector<unsigned> *to_erase,
                                bool filter_acquired_instances)
    //--------------------------------------------------------------------------
    {
      std::map<PhysicalManager*,unsigned> &already_acquired =
        *(acquired_instances);
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

    /////////////////////////////////////////////////////////////
    // Mapper Manager 
    /////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    MapperManager::MapperManager(Runtime *rt, Mapping::Mapper *mp, 
                                 MapperID mid, Processor p, bool is_default)
      : runtime(rt), mapper(mp), mapper_id(mid), processor(p),
        profile_mapper(runtime->profiler != NULL),
        request_valid_instances(mp->request_valid_instances()),
        is_default_mapper(is_default), initially_reentrant(
            mapper->get_mapper_sync_model() != 
             Mapper::SERIALIZED_NON_REENTRANT_MAPPER_MODEL)
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
    void MapperManager::invoke_handle_message(Mapper::MapperMessage *message)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo *previous = implicit_mapper_call;
      // This is subtle: in order to avoid deadlock between mapper calls either
      // to the same mapper or between mappers, all we need to check is that
      // the mapper call is still in a reentrant mode, if it is then we can do
      // the next mapper call directly, otherwise we need to defer it
      if ((previous != NULL) && !previous->reentrant)
      {
        DeferMessageArgs args(this, message->sender, message->kind, 
                              malloc(message->size), message->size,
                              message->broadcast);
        memcpy(args.message, message->message, args.size);
        runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY);
      }
      else
      {
#ifdef DEBUG_LEGION
        implicit_mapper_call = NULL;
#endif
        MappingCallInfo ctx(this, HANDLE_MESSAGE_CALL, NULL);
        mapper->handle_message(&ctx, *message);
      }
      implicit_mapper_call = previous;
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
    void MapperManager::notify_instance_deletion(PhysicalManager *manager)
    //--------------------------------------------------------------------------
    {
      MappingCallInfo *previous = implicit_mapper_call;
      // This is subtle: in order to avoid deadlock between mapper calls either
      // to the same mapper or between mappers, all we need to check is that
      // the mapper call is still in a reentrant mode, if it is then we can do
      // the next mapper call directly, otherwise we need to defer it
      if ((previous != NULL) && !previous->reentrant) 
      {
        DeferInstanceCollectionArgs args(this, manager);
        manager->add_base_resource_ref(MAPPER_REF);
        runtime->issue_runtime_meta_task(args, LG_LATENCY_DEFERRED_PRIORITY);
      }
      else
      {
#ifdef DEBUG_LEGION
        implicit_mapper_call = NULL;
#endif
        const MappingInstance instance(manager);
        MappingCallInfo ctx(this, HANDLE_INSTANCE_COLLECTION_CALL, NULL);
        mapper->handle_instance_collection(&ctx, instance);
      }
      implicit_mapper_call = previous;
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
      margs->manager->invoke_handle_message(&message);
      // Then free up the allocated memory
      free(margs->message);
    }

    //--------------------------------------------------------------------------
    /*static*/ void MapperManager::handle_deferred_collection(const void *args)
    //--------------------------------------------------------------------------
    {
      const DeferInstanceCollectionArgs *dargs = 
        (const DeferInstanceCollectionArgs*)args;
      dargs->manager->notify_instance_deletion(dargs->instance);
      if (dargs->instance->remove_base_resource_ref(MAPPER_REF))
        delete dargs->instance;
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
      else if (info->reentrant)
        REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                        "Illegal 'disable_reentrant' call performed in mapper "
                        "%s. Reentrant calls were already enabled and we do "
                        "not support nested calls to enable them.",
                        get_mapper_name())
      info->reentrant = true;
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
      else if (!info->reentrant)
        REPORT_LEGION_ERROR(ERROR_MAPPER_SYNCHRONIZATION,
                        "Illegal 'disable_reentrant' call performed in mapper "
                        "%s. Reentrant calls were already disabled and we do "
                        "not support nested calls to disable them.",
                        get_mapper_name())
      info->reentrant = false;
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
                      "Invalid mapper context passed to mapper runtime "
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
        implicit_profiler->record_runtime_call(kind, info->pause_time,
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
        implicit_profiler->record_mapper_call(mapper_id, processor, info->kind,
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
      if (allow_reentrant && !executing_call->reentrant)
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

    //--------------------------------------------------------------------------
    bool SerializingManager::is_safe_for_unbounded_pools(void)
    //--------------------------------------------------------------------------
    {
      return (allow_reentrant && permit_reentrant);
    }

    //--------------------------------------------------------------------------
    void SerializingManager::report_unsafe_allocation_in_unbounded_pool(
        const MappingCallInfo *info, Memory memory, RuntimeCallKind kind)
    //--------------------------------------------------------------------------
    {
      RUNTIME_CALL_DESCRIPTIONS(lg_runtime_calls);
      MemoryManager *manager = runtime->find_memory_manager(memory);
      if (permit_reentrant)
        REPORT_LEGION_FATAL(LEGION_FATAL_UNSAFE_ALLOCATION_WITH_UNBOUNDED_POOLS,
            "Encountered a non-permissive unbouned memory pool in memory %s "
            "while invoking %s in mapper call %s by mapper %s with reentrant "
            "mapper calls disabled. This situation can and most likely will "
            "lead to a deadlock as mapper calls needed to ensure forward "
            "progress will not be able to run while this mapper is blocked "
            "waiting for the unbounded pool allocation to finish. To work "
            "around this currently, all serializing reentrant mappers need "
            "to ensure that reentrant mapper calls are allowed while "
            "attempting to allocated in a memory containing non-permissive "
            "unbounded pools.", manager->get_name(), lg_runtime_calls[kind],
            get_mapper_call_name(info->kind), get_mapper_name()) 
      else
        REPORT_LEGION_FATAL(LEGION_FATAL_UNSAFE_ALLOCATION_WITH_UNBOUNDED_POOLS,
            "Encountered a non-permissive unbounded memory pool in memory %s "
            "while invoking %s in mapper call %s by serializing non-reentrant "
            "mapper %s. This situation can and most likely will lead to a "
            "deadlock as mapper calls needed to ensure forward progress will "
            "not be able to run while this mapper is blocked waiting for the "
            "unbounded pool allocation to finish. To work around this "
            "currently, all mappers attempting to allocate in a memory "
            "continaing non-permissive unbounded pools must use either "
            "the serializing reentrant or concurrent mapper synchronization "
            "model.", manager->get_name(), lg_runtime_calls[kind],
            get_mapper_call_name(info->kind), get_mapper_name())
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
      if (implicit_mapper_call != info)
        REPORT_LEGION_ERROR(ERROR_INVALID_MAPPER_CONTENT,
                      "Invalid mapper context passed to mapper_rt "
                      "call by mapper %s. Mapper contexts are only valid "
                      "for the mapper call to which they are passed. They "
                      "cannot be stored beyond the lifetime of the "
                      "mapper call.", mapper->get_mapper_name())
      if (profile_mapper)
        info->pause_time = Realm::Clock::current_time_in_nanoseconds();
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::resume_mapper_call(MappingCallInfo *info,
                                               RuntimeCallKind kind)
    //--------------------------------------------------------------------------
    {
      if (profile_mapper)
        implicit_profiler->record_runtime_call(kind, info->pause_time,
            Realm::Clock::current_time_in_nanoseconds());
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::finish_mapper_call(MappingCallInfo *info)
    //--------------------------------------------------------------------------
    {
      // Record our finish time when we are done
      if (profile_mapper)
        implicit_profiler->record_mapper_call(mapper_id, processor, info->kind,
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
    bool ConcurrentManager::is_safe_for_unbounded_pools(void)
    //--------------------------------------------------------------------------
    {
      return (lock_state == UNLOCKED_STATE);
    }

    //--------------------------------------------------------------------------
    void ConcurrentManager::report_unsafe_allocation_in_unbounded_pool(
        const MappingCallInfo *info, Memory memory, RuntimeCallKind kind)
    //--------------------------------------------------------------------------
    {
      RUNTIME_CALL_DESCRIPTIONS(lg_runtime_calls);
      MemoryManager *manager = runtime->find_memory_manager(memory);
      REPORT_LEGION_FATAL(LEGION_FATAL_UNSAFE_ALLOCATION_WITH_UNBOUNDED_POOLS,
            "Encountered a non-permissive unbouned memory pool in memory %s "
            "while invoking %s in mapper call %s by mapper %s while holding "
            "the concurrent mapper lock. This situation can and most likely "
            "will lead to a deadlock as mapper calls needed to ensure forward "
            "progress will not be able to run while this mapper is holding "
            "the lock and waiting for the unbounded pool allocation to "
            "finish. To work around this currently, concurrent mappers need "
            "to ensure that they are not holding the concurrent mapper lock "
            "while attempting to allocated in a memory containing "
            "non-permissive unbounded pools.", manager->get_name(), 
            lg_runtime_calls[kind], get_mapper_call_name(info->kind),
            get_mapper_name())
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
