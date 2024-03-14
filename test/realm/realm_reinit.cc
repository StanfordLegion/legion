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

// Realm test for repeated init/shutdown

#include <realm.h>
#include <realm/cmdline.h>

#include "osdep.h"

using namespace Realm;

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  EMPTY_TASK,
};

Logger log_app("app");

void empty_task(const void *args, size_t arglen,
                const void *userdata, size_t userlen, Processor p)
{
  log_app.info() << "empty task executed on processor " << p;
}

void top_level_task(const void *args, size_t arglen,
		    const void *userdata, size_t userlen, Processor p)
{
  int iteration = *static_cast<const int *>(args);

  log_app.print() << "runtime reinit test: iteration=" << iteration;
  
  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(p.kind());
  std::vector<Processor> procs(pq.begin(), pq.end());
  assert(!procs.empty());

  Processor last_proc = procs[procs.size() - 1];
  Event e = last_proc.spawn(EMPTY_TASK, 0, 0, ProfilingRequestSet());
  e.wait();

  log_app.info() << "completed successfully";
  
  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0 /*success*/);
}

int main(int argc, const char **argv)
{
  // we want to loop for a specified number of iterations, but we can
  //  only be sure to get at the command line inside the loop, so use
  //  a while instead of a for loop
  int iterations = 1;
  while(true) {
    int max_iterations = 4;

    printf("attempting iteration %d\n", iterations);

    int my_argc = argc;
    const char **my_argv = argv;
    
    Runtime rt;

    {
      bool ok = rt.init(&my_argc, (char ***)&my_argv);
      assert(ok);
    }

    {
      CommandLineParser cp;
      cp.add_option_int("-iters", max_iterations);
      bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
      assert(ok);
    }

    // TODO: Realm network modules do not generally support reinitialization
#if defined(REALM_USE_UCX) || defined(REALM_USE_GASNET1) || defined(REALM_USE_GASNETEX) || defined(REALM_USE_MPI) || defined(REALM_USE_KOKKOS)
    if(max_iterations > 1) {
      log_app.warning() << "network layers and/or kokkos do not support reinitialization - clamping iteration count to 1";
      max_iterations = 1;
    }
#endif

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
                                     EMPTY_TASK,
                                     CodeDescriptor(empty_task),
                                     ProfilingRequestSet()).external_wait();

    // collective launch of a single top level task
    rt.collective_spawn(p, TOP_LEVEL_TASK, &iterations, sizeof(iterations));

    // now sleep this thread until that shutdown actually happens
    int ret = rt.wait_for_shutdown();

    if(ret != 0) {
      fprintf(stderr, "abnormal shutdown on iteration %d\n", iterations);
      exit(1);
    }

    if(iterations++ >= max_iterations)
      break;
  }

  return 0;
}
