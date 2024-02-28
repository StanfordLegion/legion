/* Copyright 2023 Stanford University, NVIDIA Corporation
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


#ifndef __NULL_MAPPER_H__
#define __NULL_MAPPER_H__

#include "legion.h"

#include <stdlib.h>
#include <assert.h>

namespace Legion {
  namespace Mapping {

    /**
     * \class NullMapper
     * The null mapper provides an implementation of the mapper
     * interface that will assert on every single mapper call.
     * This is useful for building new mappers from scratch as
     * you an inherit from the null mapper and only implement
     * mapper calls that are invoked.
     */
    class NullMapper : public Mapper {
    public:
      NullMapper(MapperRuntime *rt, Machine machine); 
      NullMapper(const NullMapper &rhs);
      virtual ~NullMapper(void);
    public:
      NullMapper& operator=(const NullMapper &rhs);
    private:
      void report_unimplemented(const char *func_name, unsigned line) const;
    public:
      const char* get_mapper_name(void) const override;
      MapperSyncModel get_mapper_sync_model(void) const override;
    public: // Task mapping calls
      void select_task_options(const MapperContext    ctx,
                               const Task&            task,
                                     TaskOptions&     output) override;
      void premap_task(const MapperContext      ctx,
                       const Task&              task, 
                       const PremapTaskInput&   input,
                       PremapTaskOutput&        output) override;
      void slice_task(const MapperContext      ctx,
                      const Task&              task, 
                      const SliceTaskInput&    input,
                            SliceTaskOutput&   output) override;
      void map_task(const MapperContext      ctx,
                    const Task&              task,
                    const MapTaskInput&      input,
                          MapTaskOutput&     output) override;
      void replicate_task(MapperContext               ctx,
                          const Task&                 task,
                          const ReplicateTaskInput&   input,
                                ReplicateTaskOutput&  output) override;
      void select_task_variant(const MapperContext          ctx,
                               const Task&                  task,
                               const SelectVariantInput&    input,
                                     SelectVariantOutput&   output) override;
      void postmap_task(const MapperContext      ctx,
                        const Task&              task,
                        const PostMapInput&      input,
                              PostMapOutput&     output) override;
      void select_task_sources(const MapperContext        ctx,
                               const Task&                task,
                               const SelectTaskSrcInput&  input,
                                     SelectTaskSrcOutput& output) override;
      void report_profiling(const MapperContext      ctx,
                            const Task&              task,
                            const TaskProfilingInfo& input) override;
      void select_sharding_functor(
                         const MapperContext                ctx,
                         const Task&                        task,
                         const SelectShardingFunctorInput&  input,
                               SelectShardingFunctorOutput& output) override;
    public: // Inline mapping calls
      void map_inline(const MapperContext        ctx,
                      const InlineMapping&       inline_op,
                      const MapInlineInput&      input,
                            MapInlineOutput&     output) override;
      void select_inline_sources(const MapperContext        ctx,
                               const InlineMapping&         inline_op,
                               const SelectInlineSrcInput&  input,
                                     SelectInlineSrcOutput& output) override;
      void report_profiling(const MapperContext         ctx,
                            const InlineMapping&        inline_op,
                            const InlineProfilingInfo&  input) override;
    public: // Copy mapping calls
      void map_copy(const MapperContext      ctx,
                    const Copy&              copy,
                    const MapCopyInput&      input,
                          MapCopyOutput&     output) override;
      void select_copy_sources(const MapperContext          ctx,
                               const Copy&                  copy,
                               const SelectCopySrcInput&    input,
                                     SelectCopySrcOutput&   output) override;
      void report_profiling(const MapperContext      ctx,
                            const Copy&              copy,
                            const CopyProfilingInfo& input) override;
      void select_sharding_functor(
                         const MapperContext                ctx,
                         const Copy&                        copy,
                         const SelectShardingFunctorInput&  input,
                               SelectShardingFunctorOutput& output) override;
    public: // Close mapping calls
      void select_close_sources(const MapperContext        ctx,
                                const Close&               close,
                                const SelectCloseSrcInput&  input,
                                      SelectCloseSrcOutput& output) override;
      void report_profiling(const MapperContext       ctx,
                            const Close&              close,
                            const CloseProfilingInfo& input) override;
      void select_sharding_functor(
                         const MapperContext                ctx,
                         const Close&                       close,
                         const SelectShardingFunctorInput&  input,
                               SelectShardingFunctorOutput& output) override;
    public: // Acquire mapping calls
      void map_acquire(const MapperContext         ctx,
                       const Acquire&              acquire,
                       const MapAcquireInput&      input,
                             MapAcquireOutput&     output) override;
      void report_profiling(const MapperContext         ctx,
                            const Acquire&              acquire,
                            const AcquireProfilingInfo& input) override;
      void select_sharding_functor(
                         const MapperContext                ctx,
                         const Acquire&                     acquire,
                         const SelectShardingFunctorInput&  input,
                               SelectShardingFunctorOutput& output) override;
    public: // Release mapping calls
      void map_release(const MapperContext         ctx,
                       const Release&              release,
                       const MapReleaseInput&      input,
                             MapReleaseOutput&     output) override;
      void select_release_sources(const MapperContext       ctx,
                             const Release&                 release,
                             const SelectReleaseSrcInput&   input,
                                   SelectReleaseSrcOutput&  output) override;
      void report_profiling(const MapperContext         ctx,
                            const Release&              release,
                            const ReleaseProfilingInfo& input) override;
      void select_sharding_functor(
                         const MapperContext                ctx,
                         const Release&                     release,
                         const SelectShardingFunctorInput&  input,
                               SelectShardingFunctorOutput& output) override;
    public: // Partition mapping calls
      void select_partition_projection(const MapperContext  ctx,
                  const Partition&                          partition,
                  const SelectPartitionProjectionInput&     input,
                        SelectPartitionProjectionOutput&    output) override;
      void map_partition(const MapperContext        ctx,
                         const Partition&           partition,
                         const MapPartitionInput&   input,
                               MapPartitionOutput&  output) override;
      void select_partition_sources(
                           const MapperContext             ctx,
                           const Partition&                partition,
                           const SelectPartitionSrcInput&  input,
                                 SelectPartitionSrcOutput& output) override;
      void report_profiling(const MapperContext              ctx,
                            const Partition&                 partition,
                            const PartitionProfilingInfo&    input) override;
      void select_sharding_functor(
                         const MapperContext                ctx,
                         const Partition&                   partition,
                         const SelectShardingFunctorInput&  input,
                               SelectShardingFunctorOutput& output) override;
    public: // Fill mapper calls
      void select_sharding_functor(
                         const MapperContext                ctx,
                         const Fill&                        fill,
                         const SelectShardingFunctorInput&  input,
                               SelectShardingFunctorOutput& output) override;
    public:
      void map_future_map_reduction(const MapperContext     ctx,
                          const FutureMapReductionInput&    input,
                                FutureMapReductionOutput&   output) override;
    public: // Task execution mapping calls
      void configure_context(const MapperContext         ctx,
                             const Task&                 task,
                                   ContextConfigOutput&  output) override;
      void select_tunable_value(const MapperContext         ctx,
                                const Task&                 task,
                                const SelectTunableInput&   input,
                                      SelectTunableOutput&  output) override;
    public: // Must epoch mapping
      void select_sharding_functor(
                  const MapperContext                    ctx,
                  const MustEpoch&                       epoch,
                  const SelectShardingFunctorInput&      input,
                        MustEpochShardingFunctorOutput&  output) override;
      void map_must_epoch(const MapperContext           ctx,
                          const MapMustEpochInput&      input,
                                MapMustEpochOutput&     output) override;
    public: // Dataflow graph mapping
      void map_dataflow_graph(const MapperContext           ctx,
                              const MapDataflowGraphInput&  input,
                                    MapDataflowGraphOutput& output) override;
    public: // Memoization control
      void memoize_operation(const MapperContext  ctx,
                             const Mappable&      mappable,
                             const MemoizeInput&  input,
                                   MemoizeOutput& output) override;
    public: // Mapping control and stealing
      void select_tasks_to_map(const MapperContext          ctx,
                               const SelectMappingInput&    input,
                                     SelectMappingOutput&   output) override;
      void select_steal_targets(const MapperContext         ctx,
                                const SelectStealingInput&  input,
                                      SelectStealingOutput& output) override;
      void permit_steal_request(const MapperContext         ctx,
                                const StealRequestInput&    intput,
                                      StealRequestOutput&   output) override;
    public: // handling
      void handle_message(const MapperContext           ctx,
                          const MapperMessage&          message) override;
      void handle_task_result(const MapperContext           ctx,
                              const MapperTaskResult&       result) override;
      void handle_instance_collection(const MapperContext ctx,
                                      const PhysicalInstance& inst) override;
    protected:
      const Machine machine;
    };

  }; // namespace Mapping
}; // namespace Legion

#endif // __NULL_MAPPER_H___

