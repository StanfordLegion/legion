/* Copyright 2024 Stanford University, NVIDIA Corporation
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
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  READER_TASK_0,
  READER_TASK_1,
};

Logger log_app("app");

namespace ProgramConfig {
  size_t num_tasks = 1;
};

struct TaskArgs {
  int x;
};

void reader_task_0(const void *args, size_t arglen, const void *userdata,
                   size_t userlen, Processor p) {
  const TaskArgs *task_args = reinterpret_cast<const TaskArgs *>(args);
  log_app.info() << "reader task 0: proc=" << p << " x=" << task_args->x;
}

void reader_task_1(const void *args, size_t arglen, const void *userdata,
                   size_t userlen, Processor p) {
  const TaskArgs *task_args = reinterpret_cast<const TaskArgs *>(args);
  log_app.info() << "reader task 1: proc=" << p << " x=" << task_args->x;
}

void main_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) {
  TaskArgs task_args{.x = 7};

  UserEvent user_event = UserEvent::create_user_event();

  std::vector<Event> events;
  for (size_t i = 0; i < ProgramConfig::num_tasks; i++) {
    Event reader_event0 =
        p.spawn(READER_TASK_0, &task_args, sizeof(TaskArgs), user_event);
    Event reader_event1 =
        p.spawn(READER_TASK_1, &task_args, sizeof(TaskArgs), reader_event0);
    events.push_back(reader_event1);
  }

  user_event.trigger();
  Event::merge_events(events).wait();

  log_app.info() << "Completed successfully";
  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
}

int main(int argc, const char **argv) {
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  if (!p.exists()) {
    p = Machine::ProcessorQuery(Machine::get_machine()).first();
  }

  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet())
      .external_wait();

  Processor::register_task_by_kind(p.kind(), false /*!global*/, READER_TASK_0,
                                   CodeDescriptor(reader_task_0),
                                   ProfilingRequestSet())
      .external_wait();

  Processor::register_task_by_kind(p.kind(), false /*!global*/, READER_TASK_1,
                                   CodeDescriptor(reader_task_1),
                                   ProfilingRequestSet())
      .external_wait();

  rt.collective_spawn(p, MAIN_TASK, 0, 0);

  int ret = rt.wait_for_shutdown();

  return ret;
}
