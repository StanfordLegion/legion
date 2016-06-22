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

#define RUN_ALL_REGION_RULES(CALLBACK)                               \
    for (unsigned i = 0; i < region_rules.size(); ++i)               \
    {                                                                \
      bishop_region_rule_t& rule = region_rules[i];                  \
      if (rule.CALLBACK)                                             \
      {                                                              \
        legion_task_t task_ = CObjectWrapper::wrap(task);            \
        for (unsigned idx = 0; idx < task->regions.size(); ++idx)    \
        {                                                            \
          legion_region_requirement_t req_ =                         \
            CObjectWrapper::wrap(&task->regions[idx]);               \
          rule.CALLBACK(mapper_state, task_, req_, idx);             \
        }                                                            \
      }                                                              \
    }                                                                \

//------------------------------------------------------------------------------
BishopMapper::BishopMapper(const std::vector<bishop_task_rule_t>& trules,
                           const std::vector<bishop_region_rule_t>& rrules,
                   const std::vector<bishop_matching_state_transition_t>& trans,
                           bishop_mapper_state_init_fn_t init_fn,
                           MapperRuntime* rt, Machine machine,
                           Processor local_proc)
  : DefaultMapper(rt, machine, local_proc, "bishop"),
    task_rules(trules), region_rules(rrules), mapper_init(init_fn),
    runtime_(CObjectWrapper::wrap(rt))
//------------------------------------------------------------------------------
{
  log_bishop.info("bishop mapper created");
  for (unsigned idx = 0; idx < trans.size(); ++idx)
  {
    const bishop_matching_state_transition_t& t = trans[idx];
    transitions[t.state][t.task_id] = t.next_state;
  }
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

  assert(transitions.find(prev_state) != transitions.end());
  bishop_transition_func_t& trans = transitions[prev_state];
  bishop_matching_state_t curr_state = trans[task.task_id];

  DefaultMapper::select_task_options(ctx, task, output);
  legion_mapper_context_t ctx_ = CObjectWrapper::wrap(ctx);
  legion_task_t task_ = CObjectWrapper::wrap_const(&task);
  legion_task_options_t options_ = CObjectWrapper::wrap(output);
  for (unsigned i = 0; i < task_rules.size(); ++i)
  {
    bishop_task_rule_t& rule = task_rules[i];
    if (rule.select_task_options)
      rule.select_task_options(mapper_state, runtime_, ctx_, task_, options_);
  }
  output = CObjectWrapper::unwrap(options_);

  TransMsg msg;
  msg.uid = task.get_unique_id();
  msg.state = curr_state;
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
  for (unsigned i = 0; i < task_rules.size(); ++i)
  {
    bishop_task_rule_t& rule = task_rules[i];
    if (rule.map_task)
      rule.map_task(mapper_state, runtime_, ctx_, task_, input_, output_);
  }
  for (unsigned i = 0; i < region_rules.size(); ++i)
  {
    bishop_region_rule_t& rule = region_rules[i];
    if (rule.map_task)
      rule.map_task(mapper_state, runtime_, ctx_, task_, input_, output_);
  }
}

void BishopMapper::handle_message(const MapperContext  ctx,
                                  const MapperMessage& message)
{
  const TransMsg* msg = reinterpret_cast<const TransMsg*>(message.message);
  assert(message.size == sizeof(TransMsg));
  states[msg->uid] = msg->state;
}

}; // namespace Mapping

}; // namespace Legion
