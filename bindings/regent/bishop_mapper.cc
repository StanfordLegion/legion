/* Copyright 2024 Stanford University
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

#include "bishop_mapper.h"
#include "legion/legion_c_util.h"

#define LEGION_ENABLE_C_BINDINGS
#include "legion.h"

using namespace std;

namespace Legion {

namespace Mapping {

Logger log_bishop("bishop");

//------------------------------------------------------------------------------
BishopMapper::BishopMapper(const std::vector<bishop_mapper_impl_t>& impls,
                           const std::vector<bishop_transition_fn_t>& trans,
                           bishop_mapper_state_init_fn_t init_fn,
                           MapperRuntime* rt, Machine machine,
                           Processor local_proc)
  : DefaultMapper(rt, machine, local_proc, "bishop"),
    mapper_impls(impls), transitions(trans),
    mapper_init(init_fn), runtime_(CObjectWrapper::wrap(rt))
//------------------------------------------------------------------------------
{
  mapper_init(&mapper_state);
}

//------------------------------------------------------------------------------
BishopMapper::~BishopMapper()
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
void BishopMapper::select_task_options(const MapperContext ctx,
                                       const Task&         task,
                                       TaskOptions&        output)
//------------------------------------------------------------------------------
{
  log_bishop.debug("[select_task_options] start @ processor %llx",
      local_proc.id);

  bishop_matching_state_t curr_state = get_current_state(task.tag, task);

  log_bishop.debug("[select_task_options] task id %d, uid %llu, name %s, state %d",
      task.task_id, task.get_unique_id(), task.get_task_name(),
      curr_state);
  bishop_mapper_impl_t& impl = get_mapper_impl(curr_state);

  legion_mapper_context_t ctx_ = CObjectWrapper::wrap(ctx);
  legion_task_options_t options_ = CObjectWrapper::wrap(output);
  legion_task_t task_ = CObjectWrapper::wrap_const(&task);
  if (impl.select_task_options)
  {
    impl.select_task_options(mapper_state, runtime_, ctx_, task_, &options_);
    output = CObjectWrapper::unwrap(options_);
  }
  else
    DefaultMapper::select_task_options(ctx, task, output);
}

//------------------------------------------------------------------------------
void BishopMapper::slice_task(const MapperContext    ctx,
                              const Task&            task,
                              const SliceTaskInput&  input,
                              SliceTaskOutput&       output)
//------------------------------------------------------------------------------
{
  log_bishop.debug("[slice_task] start @ processor %llx", local_proc.id);
  bishop_matching_state_t curr_state = get_current_state(task.tag, task);

  legion_mapper_context_t ctx_ = CObjectWrapper::wrap(ctx);
  legion_task_t task_ = CObjectWrapper::wrap_const(&task);
  legion_slice_task_input_t input_ = CObjectWrapper::wrap_const(input);
  legion_slice_task_output_t output_ = CObjectWrapper::wrap(&output);

  bishop_mapper_impl_t& impl = get_mapper_impl(curr_state);
  if (impl.slice_task)
    impl.slice_task(mapper_state, runtime_, ctx_, task_, input_, output_);
  else
    DefaultMapper::slice_task(ctx, task, input, output);
}

//------------------------------------------------------------------------------
void BishopMapper::map_task(const MapperContext ctx,
                            const Task&         task,
                            const MapTaskInput& input,
                            MapTaskOutput&      output)
//------------------------------------------------------------------------------
{
  log_bishop.debug("[map_task] start @ processor %llx", local_proc.id);
  bishop_matching_state_t curr_state = get_current_state(task.tag, task);

  legion_mapper_context_t ctx_ = CObjectWrapper::wrap(ctx);
  legion_task_t task_ = CObjectWrapper::wrap_const(&task);
  legion_map_task_input_t input_ = CObjectWrapper::wrap_const(&input);
  legion_map_task_output_t output_ = CObjectWrapper::wrap(&output);

  log_bishop.debug("[map_task] task id %d, uid %llu, name %s, state %d",
      task.task_id, task.get_unique_id(), task.get_task_name(),
      curr_state);

  bishop_mapper_impl_t& impl = get_mapper_impl(curr_state);
  if (impl.map_task)
    impl.map_task(mapper_state, runtime_, ctx_, task_, input_, output_);
  else
    DefaultMapper::map_task(ctx, task, input, output);

  // TODO: this should be controlled from Bishop
  VariantInfo chosen = default_find_preferred_variant(task, ctx,
      true, true, output.target_procs.begin()->kind());
  output.chosen_variant = chosen.variant;
}

//------------------------------------------------------------------------------
bishop_mapper_impl_t& BishopMapper::get_mapper_impl(bishop_matching_state_t st)
//------------------------------------------------------------------------------
{
#ifdef DEBUG_LEGION
  assert(st >= 0 && st < mapper_impls.size());
#endif
  return mapper_impls[st];
}

//------------------------------------------------------------------------------
bishop_matching_state_t BishopMapper::get_current_state(
                                             bishop_matching_state_t curr_state,
                                                               const Task& task)
//------------------------------------------------------------------------------
{
  bishop_matching_state_t prev_state = curr_state;
  legion_task_t task_ = CObjectWrapper::wrap_const(&task);
  while (true)
  {
    curr_state = transitions[prev_state](task_);
    log_bishop.debug("[get_current_state] state %d --> state %d",
        prev_state, curr_state);
    if (curr_state == prev_state) break;
    prev_state = curr_state;
  }

  return curr_state;
}

}; // namespace Mapping

}; // namespace Legion
