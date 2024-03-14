/* Copyright 2024 NVIDIA Corporation
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

#include "realm.h"
#include "realm/cmdline.h"

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  Reservation reservation = Reservation::create_reservation();

  {
    UserEvent start_event = UserEvent::create_user_event();
    start_event.cancel();
    Event e1 = reservation.acquire(0, true, start_event);
    Event e2 = reservation.acquire(0, true, Event::ignorefaults(e1));
    reservation.release(e2);
    // hangs if poisoned reservation is acquired at e1 event
    reservation.acquire(0, true, e2).wait();
    reservation.release();
  }

  {
    UserEvent start_event = UserEvent::create_user_event();
    Event e1 = reservation.acquire(0, true, start_event);
    start_event.cancel();
    // fails if poisoned release is not dropped
    reservation.release(e1);
    reservation.acquire(0, true).wait();
    reservation.release();
  }

  {
    UserEvent start_event = UserEvent::create_user_event();
    Event e1 = reservation.acquire(0, true, start_event);
    Event e2 = reservation.acquire(0, true, Event::ignorefaults(e1));
    reservation.release(e2);
    start_event.cancel();
    reservation.acquire(0, true, e2).wait();
    reservation.release();
  }

  reservation.destroy_reservation();
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  CommandLineParser cp;
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task), ProfilingRequestSet(), 0, 0)
      .wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
  rt.shutdown(e);
  int ret = rt.wait_for_shutdown();
  return ret;
}
