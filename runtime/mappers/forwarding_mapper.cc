/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include "mappers/forwarding_mapper.h"

namespace Legion {
namespace Mapping {

ForwardingMapper::ForwardingMapper(Mapper* _mapper)
  : Mapper(_mapper->runtime), mapper(_mapper) {
}

ForwardingMapper::~ForwardingMapper() {
  delete mapper;
}

const char* ForwardingMapper::get_mapper_name() const {
  return mapper->get_mapper_name();
}

Mapper::MapperSyncModel ForwardingMapper::get_mapper_sync_model() const {
  return mapper->get_mapper_sync_model();
}

bool ForwardingMapper::request_valid_instances() const {
  return mapper->request_valid_instances();
}

void ForwardingMapper::select_task_options(
    const MapperContext ctx,
    const Task& task,
    TaskOptions& output) {
  mapper->select_task_options(ctx, task, output);
}

void ForwardingMapper::premap_task(
    const MapperContext ctx,
    const Task& task,
    const PremapTaskInput& input,
    PremapTaskOutput& output) {
  mapper->premap_task(ctx, task, input, output);
}

void ForwardingMapper::slice_task(
    const MapperContext ctx,
    const Task& task,
    const SliceTaskInput& input,
    SliceTaskOutput& output) {
  mapper->slice_task(ctx, task, input, output);
}

void ForwardingMapper::map_task(
    const MapperContext ctx,
    const Task& task,
    const MapTaskInput& input,
    MapTaskOutput& output) {
  mapper->map_task(ctx, task, input, output);
}

void ForwardingMapper::select_task_variant(
    const MapperContext ctx,
    const Task& task,
    const SelectVariantInput& input,
    SelectVariantOutput& output) {
  mapper->select_task_variant(ctx, task, input, output);
}

void ForwardingMapper::postmap_task(
    const MapperContext ctx,
    const Task& task,
    const PostMapInput& input,
    PostMapOutput& output) {
  mapper->postmap_task(ctx, task, input, output);
}

void ForwardingMapper::select_task_sources(
    const MapperContext ctx,
    const Task& task,
    const SelectTaskSrcInput& input,
    SelectTaskSrcOutput& output) {
  mapper->select_task_sources(ctx, task, input, output);
}

void ForwardingMapper::speculate(
    const MapperContext ctx,
    const Task& task,
    SpeculativeOutput& output) {
  mapper->speculate(ctx, task, output);
}

void ForwardingMapper::report_profiling(
    const MapperContext ctx,
    const Task& task,
    const TaskProfilingInfo& input) {
  mapper->report_profiling(ctx, task, input);
}

void ForwardingMapper::map_inline(
    const MapperContext ctx,
    const InlineMapping& inline_op,
    const MapInlineInput& input,
    MapInlineOutput& output) {
  mapper->map_inline(ctx, inline_op, input, output);
}

void ForwardingMapper::select_inline_sources(
    const MapperContext ctx,
    const InlineMapping& inline_op,
    const SelectInlineSrcInput& input,
    SelectInlineSrcOutput& output) {
  mapper->select_inline_sources(ctx, inline_op, input, output);
}

void ForwardingMapper::report_profiling(
    const MapperContext ctx,
    const InlineMapping& inline_op,
    const InlineProfilingInfo& input) {
  mapper->report_profiling(ctx, inline_op, input);
}

void ForwardingMapper::map_copy(
    const MapperContext ctx,
    const Copy& copy,
    const MapCopyInput& input,
    MapCopyOutput& output) {
  mapper->map_copy(ctx, copy, input, output);
}

void ForwardingMapper::select_copy_sources(
    const MapperContext ctx,
    const Copy& copy,
    const SelectCopySrcInput& input,
    SelectCopySrcOutput& output) {
  mapper->select_copy_sources(ctx, copy, input, output);
}

void ForwardingMapper::speculate(
    const MapperContext ctx,
    const Copy& copy,
    SpeculativeOutput& output) {
  mapper->speculate(ctx, copy, output);
}

void ForwardingMapper::report_profiling(
    const MapperContext ctx,
    const Copy& copy,
    const CopyProfilingInfo& input) {
  mapper->report_profiling(ctx, copy, input);
}

void ForwardingMapper::select_close_sources(
    const MapperContext ctx,
    const Close& close,
    const SelectCloseSrcInput& input,
    SelectCloseSrcOutput& output) {
  mapper->select_close_sources(ctx, close, input, output);
}

void ForwardingMapper::report_profiling(
    const MapperContext ctx,
    const Close& close,
    const CloseProfilingInfo& input) {
  mapper->report_profiling(ctx, close, input);
}

void ForwardingMapper::map_acquire(
    const MapperContext ctx,
    const Acquire& acquire,
    const MapAcquireInput& input,
    MapAcquireOutput& output) {
  mapper->map_acquire(ctx, acquire, input, output);
}

void ForwardingMapper::speculate(
    const MapperContext ctx,
    const Acquire& acquire,
    SpeculativeOutput& output) {
  mapper->speculate(ctx, acquire, output);
}

void ForwardingMapper::report_profiling(
    const MapperContext ctx,
    const Acquire& acquire,
    const AcquireProfilingInfo& input) {
  mapper->report_profiling(ctx, acquire, input);
}

void ForwardingMapper::map_release(
    const MapperContext ctx,
    const Release& release,
    const MapReleaseInput& input,
    MapReleaseOutput& output) {
  mapper->map_release(ctx, release, input, output);
}

void ForwardingMapper::select_release_sources(
    const MapperContext ctx,
    const Release& release,
    const SelectReleaseSrcInput& input,
    SelectReleaseSrcOutput& output) {
  mapper->select_release_sources(ctx, release, input, output);
}

void ForwardingMapper::speculate(
    const MapperContext ctx,
    const Release& release,
    SpeculativeOutput& output) {
  mapper->speculate(ctx, release, output);
}

void ForwardingMapper::report_profiling(
    const MapperContext ctx,
    const Release& release,
    const ReleaseProfilingInfo& input) {
  mapper->report_profiling(ctx, release, input);
}

void ForwardingMapper::select_partition_projection(
    const MapperContext ctx,
    const Partition& partition,
    const SelectPartitionProjectionInput& input,
    SelectPartitionProjectionOutput& output) {
  mapper->select_partition_projection(ctx, partition, input, output);
}

void ForwardingMapper::map_partition(
    const MapperContext ctx,
    const Partition& partition,
    const MapPartitionInput& input,
    MapPartitionOutput& output) {
  mapper->map_partition(ctx, partition, input, output);
}

void ForwardingMapper::select_partition_sources(
    const MapperContext ctx,
    const Partition& partition,
    const SelectPartitionSrcInput& input,
    SelectPartitionSrcOutput& output) {
  mapper->select_partition_sources(ctx, partition, input, output);
}

void ForwardingMapper::report_profiling(
    const MapperContext ctx,
    const Partition& partition,
    const PartitionProfilingInfo& input) {
  mapper->report_profiling(ctx, partition, input);
}

void ForwardingMapper::configure_context(
    const MapperContext ctx,
    const Task& task,
    ContextConfigOutput& output) {
  mapper->configure_context(ctx, task, output);
}

void ForwardingMapper::select_tunable_value(
    const MapperContext ctx,
    const Task& task,
    const SelectTunableInput& input,
    SelectTunableOutput& output) {
  mapper->select_tunable_value(ctx, task, input, output);
}

void ForwardingMapper::map_must_epoch(
    const MapperContext ctx,
    const MapMustEpochInput& input,
    MapMustEpochOutput& output) {
  mapper->map_must_epoch(ctx, input, output);
}

void ForwardingMapper::map_dataflow_graph(
    const MapperContext ctx,
    const MapDataflowGraphInput& input,
    MapDataflowGraphOutput& output) {
  mapper->map_dataflow_graph(ctx, input, output);
}

void ForwardingMapper::memoize_operation(
    const MapperContext ctx,
    const Mappable& mappable,
    const MemoizeInput& input,
    MemoizeOutput& output) {
  mapper->memoize_operation(ctx, mappable, input, output);
}

void ForwardingMapper::select_tasks_to_map(
    const MapperContext ctx,
    const SelectMappingInput& input,
    SelectMappingOutput& output) {
  mapper->select_tasks_to_map(ctx, input, output);
}

void ForwardingMapper::select_steal_targets(
    const MapperContext ctx,
    const SelectStealingInput& input,
    SelectStealingOutput& output) {
  mapper->select_steal_targets(ctx, input, output);
}

void ForwardingMapper::permit_steal_request(
    const MapperContext ctx,
    const StealRequestInput& input,
    StealRequestOutput& output) {
  mapper->permit_steal_request(ctx, input, output);
}

void ForwardingMapper::handle_message(
    const MapperContext ctx,
    const MapperMessage& message) {
  mapper->handle_message(ctx, message);
}

void ForwardingMapper::handle_task_result(
    const MapperContext ctx,
    const MapperTaskResult& result) {
  mapper->handle_task_result(ctx, result);
}

#ifndef NO_LEGION_CONTROL_REPLICATION

void ForwardingMapper::map_replicate_task(
    const MapperContext ctx,
    const Task& task,
    const MapTaskInput& input,
    const MapTaskOutput& default_output,
    MapReplicateTaskOutput& output) {
  mapper->map_replicate_task(ctx, task, input, default_output, output);
}

void ForwardingMapper::select_sharding_functor(
    const MapperContext ctx,
    const Task& task,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output) {
  mapper->select_sharding_functor(ctx, task, input, output);
}

void ForwardingMapper::select_sharding_functor(
    const MapperContext ctx,
    const Copy& copy,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output) {
  mapper->select_sharding_functor(ctx, copy, input, output);
}

void ForwardingMapper::select_sharding_functor(
    const MapperContext ctx,
    const Close& close,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output) {
  mapper->select_sharding_functor(ctx, close, input, output);
}

void ForwardingMapper::select_sharding_functor(
    const MapperContext ctx,
    const Acquire& acquire,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output) {
  mapper->select_sharding_functor(ctx, acquire, input, output);
}

void ForwardingMapper::select_sharding_functor(
    const MapperContext ctx,
    const Release& release,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output) {
  mapper->select_sharding_functor(ctx, release, input, output);
}

void ForwardingMapper::select_sharding_functor(
    const MapperContext ctx,
    const Partition& partition,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output) {
  mapper->select_sharding_functor(ctx, partition, input, output);
}

void ForwardingMapper::select_sharding_functor(
    const MapperContext ctx,
    const Fill& fill,
    const SelectShardingFunctorInput& input,
    SelectShardingFunctorOutput& output) {
  mapper->select_sharding_functor(ctx, fill, input, output);
}

void ForwardingMapper::select_sharding_functor(
    const MapperContext ctx,
    const MustEpoch& epoch,
    const SelectShardingFunctorInput& input,
    MustEpochShardingFunctorOutput& output) {
  mapper->select_sharding_functor(ctx, epoch, input, output);
}

#endif // NO_LEGION_CONTROL_REPLICATION

}; // namespace Mapping
}; // namespace Legion
