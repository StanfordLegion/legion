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

// Realm test for subscribing to events

#include <realm.h>
#include <realm/cmdline.h>
#include <realm/id.h>

#include "osdep.h"

using namespace Realm;

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  WAITER_TASK,
};

Logger log_app("app");

struct WaiterTaskArgs {
  UserEvent e;
  int expected_result; // 0: not triggered, 1: triggered
};

void waiter_task(const void *args, size_t arglen,
		 const void *userdata, size_t userlen, Processor p)
{
  WaiterTaskArgs task_args = *reinterpret_cast<const WaiterTaskArgs *>(args);
  Event e = task_args.e;
  int expected_result = task_args.expected_result;

  log_app.info() << "waiter task: proc=" << p << " event=" << e;

  if(e.has_triggered()) {
    log_app.info() << "event already triggered - nothing to do!" << e;
    assert(expected_result == 1);
    return;
  }

  // e is a remote event and we do not reuse the gen_event of e, so e's later
  //  generation should not be subscribed, thus, has_triggered should return false
  assert(expected_result == 0);

  // the event was triggered by the spawner, but we should NOT see it
  //  no matter how long we wait
  int delay = 0;
  while(delay < 1000000) {
    usleep(100000);
    delay += 100000;
    if(e.has_triggered()) {
      log_app.fatal() << "event updated without subsubscription - e=" << e << " t=" << delay << "us";
      abort();
    }
  }

  // now subscribe to the event
  e.subscribe();

  // and now expect the event's trigger to become visible fairly quickly
  delay = 0;
  while(delay < 1000000) {
    usleep(10000);
    delay += 10000;
    if(e.has_triggered()) {
      log_app.info() << "event update observed - e=" << e << " t=" << delay << "us";
      return;
    }
  }

  log_app.fatal() << "event update not observed after subscription - e=" << e << " t=" << delay << "us";
  abort();
}

void top_level_task(const void *args, size_t arglen,
		    const void *userdata, size_t userlen, Processor p)
{
  log_app.print() << "event subscription test";
  
  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(p.kind());

  UserEvent start = UserEvent::create_user_event();
  std::vector<UserEvent> begin;
  std::vector<Event> done;

  for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it) {
    // we need a UserEvent that has been triggered to give to the task, but
    //  we need to make sure that the task's own completion event isn't a later
    //  generation of the same thing, so this little dance gets us that

    // create two user events
    UserEvent e1 = UserEvent::create_user_event();
    WaiterTaskArgs args;
    args.e = e1;
    if(p.address_space() == it->address_space()) {
      args.expected_result = 1;
    } else {
      args.expected_result = 0;
    }

    // pass the first to the child task and use the second as a precondition
    Event e2 = (*it).spawn(WAITER_TASK, &args, sizeof(WaiterTaskArgs), start);
    begin.push_back(e1);
    done.push_back(e2);

    log_app.info() << "event e1:" << e1 << ", e2:" << e2 << " are created on p:" << p;
  }

  for(UserEvent &e : begin) {
    e.trigger();
  }
  start.trigger();
  Event::merge_events(done).wait();

  log_app.info() << "completed successfully";
  
  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
}

int main(int argc, const char **argv)
{
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  // try to use a cpu proc, but if that doesn't exist, take whatever we can get
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  if(!p.exists())
    p = Machine::ProcessorQuery(Machine::get_machine()).first();
  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/,
				   TOP_LEVEL_TASK,
				   CodeDescriptor(top_level_task),
				   ProfilingRequestSet()).external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/,
				   WAITER_TASK,
				   CodeDescriptor(waiter_task),
				   ProfilingRequestSet()).external_wait();

  // collective launch of a single top level task
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // now sleep this thread until that shutdown actually happens
  int ret = rt.wait_for_shutdown();
  
  return ret;
}
