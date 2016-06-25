/* Copyright 2016 Stanford University
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
#include "legion_c_util.h"
#include "utilities.h"

extern "C" {
#include "legion_c.h"
}

using namespace std;

namespace Legion {

namespace Mapping {

LegionRuntime::Logger::Category log_bishop("bishop");

//------------------------------------------------------------------------------
BishopMapper::BishopMapper(const std::vector<bishop_mapper_impl_t>& impls,
                           const std::vector<bishop_transition_fn_t>& trans,
                   const map<bishop_matching_state_t, unsigned>& state_impl_map,
                           bishop_mapper_state_init_fn_t init_fn,
                           MapperRuntime* rt, Machine machine,
                           Processor local_proc)
  : DefaultMapper(rt, machine, local_proc, "bishop"),
    mapper_impls(impls), transitions(trans),
    state_to_mapper_impl_id(state_impl_map),
    mapper_init(init_fn), runtime_(CObjectWrapper::wrap(rt))
//------------------------------------------------------------------------------
{
  log_bishop.info("bishop mapper created");
  mapper_init(&mapper_state);
}

//------------------------------------------------------------------------------
BishopMapper::~BishopMapper()
//------------------------------------------------------------------------------
{
  log_bishop.info("bishop mapper destroyed");
}

struct TransMsg
{
  UniqueID uid;
  bishop_matching_state_t state;
};

//------------------------------------------------------------------------------
void BishopMapper::select_task_options(const MapperContext ctx,
                                       const Task&         task,
                                       TaskOptions&        output)
//------------------------------------------------------------------------------
{
  bishop_matching_state_t prev_state = 0;
  std::map<UniqueID, bishop_matching_state_t>::iterator finder =
    states.find(task.parent_task->get_unique_id());
  if (finder != states.end()) prev_state = finder->second;

  bishop_transition_fn_t& trans_fn = transitions[prev_state];
  bishop_matching_state_t curr_state = 0;
  legion_task_t task_ = CObjectWrapper::wrap_const(&task);
  while (true)
  {
    curr_state = trans_fn(mapper_state, task_);
    if (curr_state == prev_state) break;
    trans_fn = transitions[curr_state];
    prev_state = curr_state;
  }

#ifdef DEBUG_LEGION
  assert(curr_state != 0);
#endif

  log_bishop.info("[select_task_options] task id %d, uid %llu, name %s, state %d",
      task.task_id, task.get_unique_id(), task.get_task_name(),
      curr_state);
  bishop_mapper_impl_t& impl = get_mapper_impl(curr_state);

  DefaultMapper::select_task_options(ctx, task, output);
  legion_mapper_context_t ctx_ = CObjectWrapper::wrap(ctx);
  legion_task_options_t options_ = CObjectWrapper::wrap(output);
  if (impl.select_task_options)
    impl.select_task_options(mapper_state, runtime_, ctx_, task_, options_);
  output = CObjectWrapper::unwrap(options_);

  TransMsg msg;
  msg.uid = task.get_unique_id();
  msg.state = curr_state;
  states[msg.uid] = msg.state;
  runtime->send_message(ctx, output.initial_proc, &msg, sizeof(TransMsg));
}

//------------------------------------------------------------------------------
void BishopMapper::slice_task(const MapperContext    ctx,
                              const Task&            task,
                              const SliceTaskInput&  input,
                              SliceTaskOutput&       output)
//------------------------------------------------------------------------------
{
  DefaultMapper::slice_task(ctx, task, input, output);
}

//------------------------------------------------------------------------------
void BishopMapper::map_task(const MapperContext ctx,
                            const Task&         task,
                            const MapTaskInput& input,
                            MapTaskOutput&      output)
//------------------------------------------------------------------------------
{
  DefaultMapper::map_task(ctx, task, input, output);
  legion_mapper_context_t ctx_ = CObjectWrapper::wrap(ctx);
  legion_task_t task_ = CObjectWrapper::wrap_const(&task);
  legion_map_task_input_t input_ = CObjectWrapper::wrap_const(&input);
  legion_map_task_output_t output_ = CObjectWrapper::wrap(&output);

  bishop_matching_state_t curr_state = 0;
  std::map<UniqueID, bishop_matching_state_t>::iterator finder =
    states.find(task.get_unique_id());
#ifdef DEBUG_LEGION
  assert(finder != states.end());
#endif
  curr_state = finder->second;

  log_bishop.info("[map_task] task id %d, uid %llu, name %s, state %d",
      task.task_id, task.get_unique_id(), task.get_task_name(),
      curr_state);

  bishop_mapper_impl_t& impl = get_mapper_impl(curr_state);
  if (impl.map_task)
    impl.map_task(mapper_state, runtime_, ctx_, task_, input_, output_);
}

void BishopMapper::handle_message(const MapperContext  ctx,
                                  const MapperMessage& message)
{
  const TransMsg* msg = reinterpret_cast<const TransMsg*>(message.message);
  assert(message.size == sizeof(TransMsg));
  // TODO: need to erase tasks from states once they finished (post_task_map)
  states[msg->uid] = msg->state;
}

//------------------------------------------------------------------------------
bishop_mapper_impl_t& BishopMapper::get_mapper_impl(bishop_matching_state_t st)
//------------------------------------------------------------------------------
{
#ifdef DEBUG_LEGION
  assert(state_to_mapper_impl_id.find(st) != state_to_mapper_impl_id.end());
#endif
  return mapper_impls[state_to_mapper_impl_id[st]];
}

}; // namespace Mapping

}; // namespace Legion
