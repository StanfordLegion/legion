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

#include "mappers/null_mapper.h"

namespace Legion {
  namespace Mapping {

    Logger log_null("null_mapper");

    //--------------------------------------------------------------------------
    NullMapper::NullMapper(MapperRuntime *rt, Machine m)
      : Mapper(rt), machine(m)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    NullMapper::NullMapper(const NullMapper &rhs)
      : Mapper(rhs.runtime), machine(rhs.machine)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    NullMapper::~NullMapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    NullMapper& NullMapper::operator=(const NullMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    void NullMapper::report_unimplemented(const char *func_name,
                                          unsigned line) const
    //--------------------------------------------------------------------------
    {
      log_null.error("Unimplemented mapper method \"%s\" in mapper %s "
         "on line %d of %s", func_name, get_mapper_name(), line, __FILE__);
      assert(false);
    }

    //--------------------------------------------------------------------------
    const char* NullMapper::get_mapper_name(void) const    
    //--------------------------------------------------------------------------
    {
      // Do this one explicitly to avoid infinite recursion
      log_null.error("Unimplemented mapper method \"get_mapper_name\"");
      assert(false);
      return NULL;
    }

    //--------------------------------------------------------------------------
    Mapper::MapperSyncModel NullMapper::get_mapper_sync_model(void) const
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
      return SERIALIZED_REENTRANT_MAPPER_MODEL;
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_task_options(const MapperContext    ctx,
                                         const Task&            task,
                                               TaskOptions&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::premap_task(const MapperContext      ctx,
                                 const Task&              task, 
                                 const PremapTaskInput&   input,
                                       PremapTaskOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::slice_task(const MapperContext      ctx,
                                const Task&              task, 
                                const SliceTaskInput&    input,
                                      SliceTaskOutput&   output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::map_task(const MapperContext      ctx,
                              const Task&              task,
                              const MapTaskInput&      input,
                                    MapTaskOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_task_variant(const MapperContext          ctx,
                                         const Task&                  task,
                                         const SelectVariantInput&    input,
                                               SelectVariantOutput&   output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::postmap_task(const MapperContext      ctx,
                                  const Task&              task,
                                  const PostMapInput&      input,
                                        PostMapOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_task_sources(const MapperContext        ctx,
                                         const Task&                task,
                                         const SelectTaskSrcInput&  input,
                                               SelectTaskSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::create_task_temporary_instance(
                                    const MapperContext              ctx,
                                    const Task&                      task,
                                    const CreateTaskTemporaryInput&  input,
                                          CreateTaskTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void NullMapper::speculate(const MapperContext      ctx,
                               const Task&              task,
                                     SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::report_profiling(const MapperContext       ctx,
                                      const Task&               task,
                                      const TaskProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::map_inline(const MapperContext        ctx,
                                const InlineMapping&       inline_op,
                                const MapInlineInput&      input,
                                      MapInlineOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_inline_sources(const MapperContext     ctx,
                                        const InlineMapping&         inline_op,
                                        const SelectInlineSrcInput&  input,
                                              SelectInlineSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::create_inline_temporary_instance(
                                  const MapperContext                ctx,
                                  const InlineMapping&               inline_op,
                                  const CreateInlineTemporaryInput&  input,
                                        CreateInlineTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);  
    }

    //--------------------------------------------------------------------------
    void NullMapper::report_profiling(const MapperContext         ctx,
                                      const InlineMapping&        inline_op,
                                      const InlineProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::map_copy(const MapperContext      ctx,
                              const Copy&              copy,
                              const MapCopyInput&      input,
                                    MapCopyOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_copy_sources(const MapperContext          ctx,
                                         const Copy&                  copy,
                                         const SelectCopySrcInput&    input,
                                               SelectCopySrcOutput&   output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::create_copy_temporary_instance(
                                  const MapperContext              ctx,
                                  const Copy&                      copy,
                                  const CreateCopyTemporaryInput&  input,
                                        CreateCopyTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void NullMapper::speculate(const MapperContext      ctx,
                               const Copy&              copy,
                                     SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::report_profiling(const MapperContext      ctx,
                                      const Copy&              copy,
                                      const CopyProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }
    
    //--------------------------------------------------------------------------
    void NullMapper::map_close(const MapperContext       ctx,
                               const Close&              close,
                               const MapCloseInput&      input,
                                     MapCloseOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_close_sources(const MapperContext        ctx,
                                          const Close&               close,
                                          const SelectCloseSrcInput&  input,
                                                SelectCloseSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::create_close_temporary_instance(
                                  const MapperContext               ctx,
                                  const Close&                      close,
                                  const CreateCloseTemporaryInput&  input,
                                        CreateCloseTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void NullMapper::report_profiling(const MapperContext       ctx,
                                      const Close&              close,
                                      const CloseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::map_acquire(const MapperContext         ctx,
                                 const Acquire&              acquire,
                                 const MapAcquireInput&      input,
                                       MapAcquireOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::speculate(const MapperContext         ctx,
                               const Acquire&              acquire,
                                     SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::report_profiling(const MapperContext         ctx,
                                      const Acquire&              acquire,
                                      const AcquireProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::map_release(const MapperContext         ctx,
                                 const Release&              release,
                                 const MapReleaseInput&      input,
                                       MapReleaseOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_release_sources(const MapperContext      ctx,
                                        const Release&                 release,
                                        const SelectReleaseSrcInput&   input,
                                              SelectReleaseSrcOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::speculate(const MapperContext         ctx,
                               const Release&              release,
                                     SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::create_release_temporary_instance(
                                   const MapperContext                 ctx,
                                   const Release&                      release,
                                   const CreateReleaseTemporaryInput&  input,
                                         CreateReleaseTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::report_profiling(const MapperContext         ctx,
                                      const Release&              release,
                                      const ReleaseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_partition_projection(const MapperContext  ctx,
                        const Partition&                          partition,
                        const SelectPartitionProjectionInput&     input,
                              SelectPartitionProjectionOutput&    output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::map_partition(const MapperContext        ctx,
                               const Partition&           partition,
                               const MapPartitionInput&   input,
                                     MapPartitionOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_partition_sources(
                                     const MapperContext             ctx,
                                     const Partition&                partition,
                                     const SelectPartitionSrcInput&  input,
                                           SelectPartitionSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::create_partition_temporary_instance(
                            const MapperContext                   ctx,
                            const Partition&                      partition,
                            const CreatePartitionTemporaryInput&  input,
                                  CreatePartitionTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::report_profiling(const MapperContext              ctx,
                                    const Partition&                 partition,
                                    const PartitionProfilingInfo&    input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::configure_context(const MapperContext         ctx,
                                       const Task&                 task,
                                             ContextConfigOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_tunable_value(const MapperContext         ctx,
                                          const Task&                 task,
                                          const SelectTunableInput&   input,
                                                SelectTunableOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::map_must_epoch(const MapperContext           ctx,
                                    const MapMustEpochInput&      input,
                                          MapMustEpochOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void NullMapper::map_dataflow_graph(const MapperContext           ctx,
                                        const MapDataflowGraphInput&  input,
                                              MapDataflowGraphOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::memoize_operation(const MapperContext  ctx,
                                       const Mappable&      mappable,
                                       const MemoizeInput&  input,
                                             MemoizeOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_tasks_to_map(const MapperContext          ctx,
                                         const SelectMappingInput&    input,
                                               SelectMappingOutput&   output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void NullMapper::select_steal_targets(const MapperContext         ctx,
                                          const SelectStealingInput&  input,
                                                SelectStealingOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::permit_steal_request(const MapperContext         ctx,
                                          const StealRequestInput&    input,
                                                StealRequestOutput&   output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void NullMapper::handle_message(const MapperContext           ctx,
                                    const MapperMessage&          message)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void NullMapper::handle_task_result(const MapperContext           ctx,
                                        const MapperTaskResult&       result)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

  }; // namespace Mapping 
}; // namespace Legion

// EOF

