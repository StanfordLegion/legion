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

#ifndef __BISHOP_MAPPER_H__
#define __BISHOP_MAPPER_H__

#include <vector>
#include <map>
#include <string>

#define LEGION_ENABLE_C_BINDINGS
#include "legion.h"
#include "default_mapper.h"

extern "C" {
#include "bishop_c.h"
}

namespace Legion {

  namespace Mapping {

    class BishopMapper : public DefaultMapper
    {
      public:
        BishopMapper(const std::vector<bishop_mapper_impl_t>&,
                     const std::vector<bishop_transition_fn_t>&,
                     bishop_mapper_state_init_fn_t,
                     MapperRuntime*, Machine, Processor);
        virtual ~BishopMapper();

        virtual void select_task_options(const MapperContext ctx,
                                         const Task&         task,
                                               TaskOptions&  output);
        virtual void slice_task(const MapperContext    ctx,
                                const Task&            task,
                                const SliceTaskInput&  input,
                                      SliceTaskOutput& output);
        virtual void map_task(const MapperContext  ctx,
                              const Task&          task,
                              const MapTaskInput&  input,
                                    MapTaskOutput& output);

        bishop_mapper_impl_t& get_mapper_impl(bishop_matching_state_t state);

        bishop_matching_state_t get_current_state(bishop_matching_state_t prev_state,
                                                  const Task& task);
      private:
        std::vector<bishop_mapper_impl_t> mapper_impls;
        std::vector<bishop_transition_fn_t> transitions;

        bishop_mapper_state_init_fn_t mapper_init;
        bishop_mapper_state_t mapper_state;
        legion_mapper_runtime_t runtime_;
    };

  };

};
#endif // __BISHOP_MAPPER_H__
