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
  WRITER_TASK,
};

Logger log_app("app");

int x;

void reader_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p)
{
  log_app.info() << "reader task: proc=" << p;
  assert(x == 8);
}

void writer_task(const void *args, size_t arglen, const void *userdata,
                 size_t userlen, Processor p)
{
  log_app.info() << "writer task: proc=" << p;
  x += 7;
}

void top_level_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p)
{
  x = 1;

  // TODO(apryakhin@): Add documentation.
  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(p.kind());
  for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it) {

    UserEvent event1 = UserEvent::create_user_event();

    Event event2 = (*it).spawn(WRITER_TASK, 0, 0, event1);
    Event event3 = (*it).spawn(READER_TASK, 0, 0, event2);

    event1.trigger();
    event3.wait();
  }

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

  Processor::register_task_by_kind(p.kind(), false /*!global*/, WRITER_TASK,
                                   CodeDescriptor(writer_task),
                                   ProfilingRequestSet())
      .external_wait();

  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  int ret = rt.wait_for_shutdown();

  return ret;
}
