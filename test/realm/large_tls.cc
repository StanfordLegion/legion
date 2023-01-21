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

// Realm test for creating threads with a large amount of static TLS

#include <realm.h>
#include <realm/cmdline.h>

#include "osdep.h"

using namespace Realm;

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  WAITER_TASK,
};

Logger log_app("app");

struct LargeStruct {
  char data[256000];
};

REALM_THREAD_LOCAL LargeStruct large_tls_var;

void top_level_task(const void *args, size_t arglen,
		    const void *userdata, size_t userlen, Processor p)
{
  int stack_var;

  log_app.print() << "task stack top: " << (void *)&stack_var;
  log_app.print() << "task tls base: " << (void *)(large_tls_var.data);
#ifndef _MSC_VER
  log_app.print() << "pthread_stack_min: " << size_t(PTHREAD_STACK_MIN);
#endif

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

  // collective launch of a single top level task
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // now sleep this thread until that shutdown actually happens
  int ret = rt.wait_for_shutdown();
  
  return ret;
}
