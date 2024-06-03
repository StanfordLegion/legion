
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
#include "realm/network.h"

#ifdef REALM_USE_OPENMP
#include <omp.h>
#endif

using namespace Realm;

Logger log_app("app");

enum {
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  HELLO_TASK,
};

void hello_cpu_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) 
{
  log_app.print() << "Hello world from CPU!";
}

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
void hello_gpu_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) 
{
  log_app.print() << "Hello world from GPU!";
}
#endif

#ifdef REALM_USE_OPENMP
void hello_omp_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) 
{
  #pragma omp parallel                   
  {
    log_app.print() << "Hello world from OMP thread = " << omp_get_thread_num();
  }
}
#endif

inline Event launch_task(Processor cpu)
{
  // launch a hello task on CPU
  Event cpu_e = cpu.spawn(HELLO_TASK, NULL, 0);

  // launch a hello task on GPU if it is available
  Processor gpu = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::TOC_PROC).first();
  Event gpu_e = Event::NO_EVENT;
  if (gpu.exists()) {
    gpu_e = gpu.spawn(HELLO_TASK, NULL, 0);
  }

  // launch a hello task on OpenMP processor if it is available
  Processor omp = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::OMP_PROC).first();
  Event omp_e = Event::NO_EVENT;
  if (omp.exists()) {
    omp_e = omp.spawn(HELLO_TASK, NULL, 0);
  }

  Event e = Event::merge_events(cpu_e, gpu_e, omp_e);
  return e;
}

void main_task(const void *args, size_t arglen, const void *userdata,
               size_t userlen, Processor p) 
{
  launch_task(p).wait();
}

int main(int argc, char **argv) 
{
  Runtime rt;
  rt.init(&argc, &argv);

  bool use_collective_spawn = false;
  CommandLineParser cp;
  cp.add_option_bool("-coll_spawn", use_collective_spawn);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);
  
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   MAIN_TASK,
                                   CodeDescriptor(main_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   HELLO_TASK,
                                   CodeDescriptor(hello_cpu_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   HELLO_TASK,
                                   CodeDescriptor(hello_gpu_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
#endif
#ifdef REALM_USE_OPENMP
  Processor::register_task_by_kind(Processor::OMP_PROC, false /*!global*/,
                                   HELLO_TASK,
                                   CodeDescriptor(hello_omp_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
#endif

  // select a processor to run the cpu task
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC)
    .first();
  assert(p.exists());

  if (use_collective_spawn) {
    Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
    rt.shutdown(e);
  } else {
    // try to get the rank ID
    Processor local_proc = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::LOC_PROC).local_address_space()
    .first();
    if (local_proc.address_space() == 0) {
      Event e = launch_task(p);
      rt.shutdown(e);
    }
    // call shutdown(e) here is wrong
  }

  int ret = rt.wait_for_shutdown();

  return ret;
}
