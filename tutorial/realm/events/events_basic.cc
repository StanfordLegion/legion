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

#include <realm.h>
#include <realm/cmdline.h>

using namespace Realm;

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  READER_TASK,
};

Logger log_app("app");

namespace ProgramConfig {
  size_t num_tasks = 2;
};

void reader_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p)
{
  int x = *reinterpret_cast<const int *>(args);
  log_app.info() << "reader task: proc=" << p << " x=" << x;
}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p)
{
  int x = 7;

  std::vector<Event> events;
  for(size_t i = 0; i < ProgramConfig::num_tasks; i++) {
    UserEvent event1 = UserEvent::create_user_event();

    Event task_event = p.spawn(READER_TASK, &x, sizeof(int), event1);

    events.push_back(task_event);
    event1.trigger();
  }

  Event::merge_events(events).wait();

  log_app.info() << "Completed successfully";

  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
}

int main(int argc, const char **argv)
{
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  if(!p.exists()) {
    p = Machine::ProcessorQuery(Machine::get_machine()).first();
  }

  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/, TOP_LEVEL_TASK,
                                   CodeDescriptor(top_level_task),
                                   ProfilingRequestSet())
      .external_wait();

  Processor::register_task_by_kind(p.kind(), false /*!global*/, READER_TASK,
                                   CodeDescriptor(reader_task),
                                   ProfilingRequestSet())
      .external_wait();

  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  int ret = rt.wait_for_shutdown();

  return ret;
}
