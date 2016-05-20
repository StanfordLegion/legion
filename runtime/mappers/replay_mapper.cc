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

#include "replay_mapper.h"

namespace Legion {
  namespace Mapping {

    //--------------------------------------------------------------------------
    /*static*/ const char* ReplayMapper::create_replay_name(Processor p)
    //--------------------------------------------------------------------------
    {
      const size_t buffer_size = 64;
      char *result = (char*)malloc(buffer_size*sizeof(char));
      snprintf(result, buffer_size-1,
                "Replay Mapper on Processor " IDFMT "", p.id);
      return result;
    }

    //--------------------------------------------------------------------------
    ReplayMapper::ReplayMapper(MapperRuntime *rt, Machine m, Processor local,
                               const char *replay_file, const char *name)
      : Mapper(rt), machine(m), local_proc(local), 
        mapper_name((name == NULL) ? create_replay_name(local) : name)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ReplayMapper::ReplayMapper(const ReplayMapper &rhs)
      : Mapper(rhs.runtime), machine(rhs.machine), 
        local_proc(rhs.local_proc), mapper_name(rhs.mapper_name)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
    }

    //--------------------------------------------------------------------------
    ReplayMapper::~ReplayMapper(void)
    //--------------------------------------------------------------------------
    {
      free(const_cast<char*>(mapper_name));
    }

    //--------------------------------------------------------------------------
    ReplayMapper& ReplayMapper::operator=(const ReplayMapper &rhs)
    //--------------------------------------------------------------------------
    {
      // should never be called
      assert(false);
      return *this;
    }

    //--------------------------------------------------------------------------
    const char* ReplayMapper::get_mapper_name(void) const    
    //--------------------------------------------------------------------------
    {
      return mapper_name;
    }

    //--------------------------------------------------------------------------
    Mapper::MapperSyncModel ReplayMapper::get_mapper_sync_model(void) const
    //--------------------------------------------------------------------------
    {
      return SERIALIZED_REENTRANT_MAPPER_MODEL;
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_task_options(const MapperContext    ctx,
                                           const Task&            task,
                                                 TaskOptions&     output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::premap_task(const MapperContext      ctx,
                                   const Task&              task, 
                                   const PremapTaskInput&   input,
                                         PremapTaskOutput&  output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::slice_task(const MapperContext      ctx,
                                  const Task&              task, 
                                  const SliceTaskInput&    input,
                                        SliceTaskOutput&   output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_task(const MapperContext      ctx,
                                const Task&              task,
                                const MapTaskInput&      input,
                                      MapTaskOutput&     output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_task_variant(const MapperContext          ctx,
                                           const Task&                  task,
                                           const SelectVariantInput&    input,
                                                 SelectVariantOutput&   output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::postmap_task(const MapperContext      ctx,
                                    const Task&              task,
                                    const PostMapInput&      input,
                                          PostMapOutput&     output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_task_sources(const MapperContext        ctx,
                                           const Task&                task,
                                           const SelectTaskSrcInput&  input,
                                                 SelectTaskSrcOutput& output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::speculate(const MapperContext      ctx,
                                 const Task&              task,
                                       SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext       ctx,
                                        const Task&               task,
                                        const TaskProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_inline(const MapperContext        ctx,
                                  const InlineMapping&       inline_op,
                                  const MapInlineInput&      input,
                                        MapInlineOutput&     output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_inline_sources(const MapperContext     ctx,
                                        const InlineMapping&         inline_op,
                                        const SelectInlineSrcInput&  input,
                                              SelectInlineSrcOutput& output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext         ctx,
                                        const InlineMapping&        inline_op,
                                        const InlineProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_copy(const MapperContext      ctx,
                                const Copy&              copy,
                                const MapCopyInput&      input,
                                      MapCopyOutput&     output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_copy_sources(const MapperContext          ctx,
                                           const Copy&                  copy,
                                           const SelectCopySrcInput&    input,
                                                 SelectCopySrcOutput&   output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::speculate(const MapperContext      ctx,
                                 const Copy& copy,
                                       SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext      ctx,
                                        const Copy&              copy,
                                        const CopyProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
    }
    
    //--------------------------------------------------------------------------
    void ReplayMapper::map_close(const MapperContext       ctx,
                                 const Close&              close,
                                 const MapCloseInput&      input,
                                       MapCloseOutput&     output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_close_sources(const MapperContext        ctx,
                                            const Close&               close,
                                            const SelectCloseSrcInput&  input,
                                                  SelectCloseSrcOutput& output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext       ctx,
                                        const Close&              close,
                                        const CloseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_acquire(const MapperContext         ctx,
                                   const Acquire&              acquire,
                                   const MapAcquireInput&      input,
                                         MapAcquireOutput&     output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::speculate(const MapperContext         ctx,
                                 const Acquire&              acquire,
                                       SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext         ctx,
                                        const Acquire&              acquire,
                                        const AcquireProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_release(const MapperContext         ctx,
                                   const Release&              release,
                                   const MapReleaseInput&      input,
                                         MapReleaseOutput&     output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_release_sources(const MapperContext      ctx,
                                        const Release&                 release,
                                        const SelectReleaseSrcInput&   input,
                                              SelectReleaseSrcOutput&  output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::speculate(const MapperContext         ctx,
                                 const Release&              release,
                                       SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::report_profiling(const MapperContext         ctx,
                                        const Release&              release,
                                        const ReleaseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::configure_context(const MapperContext         ctx,
                                         const Task&                 task,
                                               ContextConfigOutput&  output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_tunable_value(const MapperContext         ctx,
                                            const Task&                 task,
                                            const SelectTunableInput&   input,
                                                  SelectTunableOutput&  output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_must_epoch(const MapperContext           ctx,
                                      const MapMustEpochInput&      input,
                                            MapMustEpochOutput&     output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::map_dataflow_graph(const MapperContext           ctx,
                                          const MapDataflowGraphInput&  input,
                                                MapDataflowGraphOutput& output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_tasks_to_map(const MapperContext          ctx,
                                           const SelectMappingInput&    input,
                                                 SelectMappingOutput&   output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::select_steal_targets(const MapperContext         ctx,
                                            const SelectStealingInput&  input,
                                                  SelectStealingOutput& output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::permit_steal_request(const MapperContext         ctx,
                                            const StealRequestInput&    input,
                                                  StealRequestOutput&   output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::handle_message(const MapperContext           ctx,
                                      const MapperMessage&          message)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ReplayMapper::handle_task_result(const MapperContext           ctx,
                                          const MapperTaskResult&       result)
    //--------------------------------------------------------------------------
    {
    }

  }; // namespace Mapping 
}; // namespace Legion

// EOF

