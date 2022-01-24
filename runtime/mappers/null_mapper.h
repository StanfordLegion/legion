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
      virtual const char* get_mapper_name(void) const;
      virtual MapperSyncModel get_mapper_sync_model(void) const;
    public: // Task mapping calls
      virtual void select_task_options(const MapperContext    ctx,
                                       const Task&            task,
                                             TaskOptions&     output);
      virtual void premap_task(const MapperContext      ctx,
                               const Task&              task, 
                               const PremapTaskInput&   input,
                               PremapTaskOutput&        output);
      virtual void slice_task(const MapperContext      ctx,
                              const Task&              task, 
                              const SliceTaskInput&    input,
                                    SliceTaskOutput&   output);
      virtual void map_task(const MapperContext      ctx,
                            const Task&              task,
                            const MapTaskInput&      input,
                                  MapTaskOutput&     output);
      virtual void select_task_variant(const MapperContext          ctx,
                                       const Task&                  task,
                                       const SelectVariantInput&    input,
                                             SelectVariantOutput&   output);
      virtual void postmap_task(const MapperContext      ctx,
                                const Task&              task,
                                const PostMapInput&      input,
                                      PostMapOutput&     output);
      virtual void select_task_sources(const MapperContext        ctx,
                                       const Task&                task,
                                       const SelectTaskSrcInput&  input,
                                             SelectTaskSrcOutput& output);
      virtual void create_task_temporary_instance(
                                    const MapperContext              ctx,
                                    const Task&                      task,
                                    const CreateTaskTemporaryInput&  input,
                                          CreateTaskTemporaryOutput& output);
      virtual void speculate(const MapperContext      ctx,
                             const Task&              task,
                                   SpeculativeOutput& output);
      virtual void report_profiling(const MapperContext      ctx,
                                    const Task&              task,
                                    const TaskProfilingInfo& input);
    public: // Inline mapping calls
      virtual void map_inline(const MapperContext        ctx,
                              const InlineMapping&       inline_op,
                              const MapInlineInput&      input,
                                    MapInlineOutput&     output);
      virtual void select_inline_sources(const MapperContext        ctx,
                                       const InlineMapping&         inline_op,
                                       const SelectInlineSrcInput&  input,
                                             SelectInlineSrcOutput& output);
      virtual void create_inline_temporary_instance(
                                  const MapperContext                ctx,
                                  const InlineMapping&               inline_op,
                                  const CreateInlineTemporaryInput&  input,
                                        CreateInlineTemporaryOutput& output);
      virtual void report_profiling(const MapperContext         ctx,
                                    const InlineMapping&        inline_op,
                                    const InlineProfilingInfo&  input);
    public: // Copy mapping calls
      virtual void map_copy(const MapperContext      ctx,
                            const Copy&              copy,
                            const MapCopyInput&      input,
                                  MapCopyOutput&     output);
      virtual void select_copy_sources(const MapperContext          ctx,
                                       const Copy&                  copy,
                                       const SelectCopySrcInput&    input,
                                             SelectCopySrcOutput&   output);
      virtual void create_copy_temporary_instance(
                                  const MapperContext              ctx,
                                  const Copy&                      copy,
                                  const CreateCopyTemporaryInput&  input,
                                        CreateCopyTemporaryOutput& output);
      virtual void speculate(const MapperContext      ctx,
                             const Copy& copy,
                                   SpeculativeOutput& output);
      virtual void report_profiling(const MapperContext      ctx,
                                    const Copy&              copy,
                                    const CopyProfilingInfo& input);
    public: // Close mapping calls
      virtual void map_close(const MapperContext       ctx,
                             const Close&              close,
                             const MapCloseInput&      input,
                                   MapCloseOutput&     output);
      virtual void select_close_sources(const MapperContext        ctx,
                                        const Close&               close,
                                        const SelectCloseSrcInput&  input,
                                              SelectCloseSrcOutput& output);
      virtual void create_close_temporary_instance(
                                  const MapperContext               ctx,
                                  const Close&                      close,
                                  const CreateCloseTemporaryInput&  input,
                                        CreateCloseTemporaryOutput& output);
      virtual void report_profiling(const MapperContext       ctx,
                                    const Close&              close,
                                    const CloseProfilingInfo& input);
    public: // Acquire mapping calls
      virtual void map_acquire(const MapperContext         ctx,
                               const Acquire&              acquire,
                               const MapAcquireInput&      input,
                                     MapAcquireOutput&     output);
      virtual void speculate(const MapperContext         ctx,
                             const Acquire&              acquire,
                                   SpeculativeOutput&    output);
      virtual void report_profiling(const MapperContext         ctx,
                                    const Acquire&              acquire,
                                    const AcquireProfilingInfo& input);
    public: // Release mapping calls
      virtual void map_release(const MapperContext         ctx,
                               const Release&              release,
                               const MapReleaseInput&      input,
                                     MapReleaseOutput&     output);
      virtual void select_release_sources(const MapperContext       ctx,
                                     const Release&                 release,
                                     const SelectReleaseSrcInput&   input,
                                           SelectReleaseSrcOutput&  output);
      virtual void create_release_temporary_instance(
                                   const MapperContext                 ctx,
                                   const Release&                      release,
                                   const CreateReleaseTemporaryInput&  input,
                                         CreateReleaseTemporaryOutput& output);
      virtual void speculate(const MapperContext         ctx,
                             const Release&              release,
                                   SpeculativeOutput&    output);
      virtual void report_profiling(const MapperContext         ctx,
                                    const Release&              release,
                                    const ReleaseProfilingInfo& input);
    public: // Partition mapping calls
      virtual void select_partition_projection(const MapperContext  ctx,
                          const Partition&                          partition,
                          const SelectPartitionProjectionInput&     input,
                                SelectPartitionProjectionOutput&    output);
      virtual void map_partition(const MapperContext        ctx,
                                 const Partition&           partition,
                                 const MapPartitionInput&   input,
                                       MapPartitionOutput&  output);
      virtual void select_partition_sources(
                                   const MapperContext             ctx,
                                   const Partition&                partition,
                                   const SelectPartitionSrcInput&  input,
                                         SelectPartitionSrcOutput& output);
      virtual void create_partition_temporary_instance(
                              const MapperContext                   ctx,
                              const Partition&                      partition,
                              const CreatePartitionTemporaryInput&  input,
                                    CreatePartitionTemporaryOutput& output);
      virtual void report_profiling(const MapperContext              ctx,
                                    const Partition&                 partition,
                                    const PartitionProfilingInfo&    input);
    public: // Task execution mapping calls
      virtual void configure_context(const MapperContext         ctx,
                                     const Task&                 task,
                                           ContextConfigOutput&  output);
      virtual void select_tunable_value(const MapperContext         ctx,
                                        const Task&                 task,
                                        const SelectTunableInput&   input,
                                              SelectTunableOutput&  output);
    public: // Must epoch mapping
      virtual void map_must_epoch(const MapperContext           ctx,
                                  const MapMustEpochInput&      input,
                                        MapMustEpochOutput&     output);
    public: // Dataflow graph mapping
      virtual void map_dataflow_graph(const MapperContext           ctx,
                                      const MapDataflowGraphInput&  input,
                                            MapDataflowGraphOutput& output);
    public: // Memoization control
      virtual void memoize_operation(const MapperContext  ctx,
                                     const Mappable&      mappable,
                                     const MemoizeInput&  input,
                                           MemoizeOutput& output);
    public: // Mapping control and stealing
      virtual void select_tasks_to_map(const MapperContext          ctx,
                                       const SelectMappingInput&    input,
                                             SelectMappingOutput&   output);
      virtual void select_steal_targets(const MapperContext         ctx,
                                        const SelectStealingInput&  input,
                                              SelectStealingOutput& output);
      virtual void permit_steal_request(const MapperContext         ctx,
                                        const StealRequestInput&    intput,
                                              StealRequestOutput&   output);
    public: // handling
      virtual void handle_message(const MapperContext           ctx,
                                  const MapperMessage&          message);
      virtual void handle_task_result(const MapperContext           ctx,
                                      const MapperTaskResult&       result);
    protected:
      const Machine machine;
    };

  }; // namespace Mapping
}; // namespace Legion

#endif // __NULL_MAPPER_H___

