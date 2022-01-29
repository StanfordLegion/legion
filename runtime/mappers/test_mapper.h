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


#ifndef __TEST_MAPPER_H__
#define __TEST_MAPPER_H__

#include "legion.h"
#include "mappers/default_mapper.h"

#include <stdlib.h>
#include <assert.h>
#include <algorithm>

namespace Legion {
  namespace Mapping {

    /**
     * \class TestMapper
     * The test mapper provides support for stress testing
     * the Legion and Realm runtime systems. The test mapper
     * will make random mapping decisions designed to exercise
     * corner cases in the runtime that might not normally
     * be explored by normal mapping decisions.
     */
    class TestMapper : public DefaultMapper {
    public:
      TestMapper(MapperRuntime *rt, Machine machine, Processor local, 
                 const char *mapper_name = NULL);
      TestMapper(const TestMapper &rhs);
      virtual ~TestMapper(void);
    public:
      TestMapper& operator=(const TestMapper &rhs);
    public:
      static const char* create_test_name(Processor p);
    public: // Task mapping calls
      virtual void select_task_options(const MapperContext    ctx,
                                       const Task&            task,
                                             TaskOptions&     output);
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
      virtual void select_task_sources(const MapperContext        ctx,
                                       const Task&                task,
                                       const SelectTaskSrcInput&  input,
                                             SelectTaskSrcOutput& output);
      virtual void speculate(const MapperContext      ctx,
                             const Task&              task,
                                   SpeculativeOutput& output);
    public: // Inline mapping calls
      virtual void map_inline(const MapperContext        ctx,
                              const InlineMapping&       inline_op,
                              const MapInlineInput&      input,
                                    MapInlineOutput&     output);
      virtual void select_inline_sources(const MapperContext        ctx,
                                       const InlineMapping&         inline_op,
                                       const SelectInlineSrcInput&  input,
                                             SelectInlineSrcOutput& output);
    public: // Copy mapping calls
      virtual void map_copy(const MapperContext      ctx,
                            const Copy&              copy,
                            const MapCopyInput&      input,
                                  MapCopyOutput&     output);
      virtual void select_copy_sources(const MapperContext          ctx,
                                       const Copy&                  copy,
                                       const SelectCopySrcInput&    input,
                                             SelectCopySrcOutput&   output);
      virtual void speculate(const MapperContext      ctx,
                             const Copy& copy,
                                   SpeculativeOutput& output);
    public: // Close mapping calls
      virtual void map_close(const MapperContext       ctx,
                             const Close&              close,
                             const MapCloseInput&      input,
                                   MapCloseOutput&     output);
      virtual void select_close_sources(const MapperContext        ctx,
                                        const Close&               close,
                                        const SelectCloseSrcInput&  input,
                                              SelectCloseSrcOutput& output);
    public: // Acquire mapping calls
      virtual void speculate(const MapperContext         ctx,
                             const Acquire&              acquire,
                                   SpeculativeOutput&    output);
    public: // Release mapping calls
      virtual void select_release_sources(const MapperContext       ctx,
                                     const Release&                 release,
                                     const SelectReleaseSrcInput&   input,
                                           SelectReleaseSrcOutput&  output);
      virtual void speculate(const MapperContext         ctx,
                             const Release&              release,
                                   SpeculativeOutput&    output);
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
    protected:
      Processor select_random_processor(Processor::Kind kind);
      Processor::Kind select_random_processor_kind(MapperContext ctx, 
                                                   TaskID task_id);
      void find_task_processor_kinds(MapperContext ctx, TaskID task_id,
                                     std::set<Processor::Kind> &kinds);
      const std::map<VariantID,Processor::Kind>&
                 find_task_variants(MapperContext ctx, TaskID task_id);
      void select_random_source_order(
          const std::vector<PhysicalInstance> &sources,
                std::deque<PhysicalInstance> &ranking);
    protected:
      void map_constrained_requirement(MapperContext ctx, 
          const RegionRequirement &req, MappingKind mapping_kind,
          const std::vector<LayoutConstraintID> &constraints,
          std::vector<PhysicalInstance> &chosen_instances,
          Processor target_proc = Processor::NO_PROC);
      void map_random_requirement(MapperContext ctx,
          const RegionRequirement &req,
          std::vector<PhysicalInstance> &chosen_instances,
          Processor target_proc = Processor::NO_PROC);
    protected:
      long generate_random_integer(void) const 
        { return default_generate_random_integer(); }
      double generate_random_real(void) const
        { return default_generate_random_real(); }
    protected:
      std::map<TaskID,std::map<VariantID,
                               Processor::Kind> > variant_processor_kinds;
    };

  }; // namespace Mapping
}; // namespace Legion

#endif // _TEST_MAPPER_H___

