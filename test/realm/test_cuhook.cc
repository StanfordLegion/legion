
/* Copyright 2023 NVIDIA Corporation
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

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_module.h"
#include <cuda_runtime.h>
#else
#include "realm/hip/hip_module.h"
#include <hip/hip_runtime.h>
#endif

#include <assert.h>

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  GPU_TASK,
  GPU_TASK_WITH_STREAM,
};

void gpu_kernel_wrapper(cudaStream_t stream = 0);

void gpu_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
              Processor p)
{
  log_app.print() << "Hello from GPU task";
  void *ptr = NULL;
  cudaMalloc(&ptr, sizeof(1024));
  cudaFree(ptr);
  cudaStream_t stream;

#ifdef REALM_USE_CUDART_HIJACK
  cudaStreamCreate(&stream);
#else
  stream = Cuda::get_task_cuda_stream();
  Cuda::set_task_ctxsync_required(false);
#endif
  gpu_kernel_wrapper(stream);
  // cudaStreamSynchronize(stream);
}

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  std::vector<Processor> gpu_procs;
  {
    Machine::ProcessorQuery query(Machine::get_machine());
    query.only_kind(Processor::TOC_PROC);
    gpu_procs.insert(gpu_procs.end(), query.begin(), query.end());
  }
  std::vector<Event> events(gpu_procs.size(), Event::NO_EVENT);
  for(size_t i = 0; i < gpu_procs.size(); i++) {
    events[i] = gpu_procs[i].spawn(GPU_TASK, NULL, 0);
  }
  Event::merge_events(events).wait();
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, TOP_LEVEL_TASK,
                                   CodeDescriptor(top_level_task), ProfilingRequestSet(),
                                   0, 0)
      .wait();
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, GPU_TASK,
                                   CodeDescriptor(gpu_task), ProfilingRequestSet(), 0, 0)
      .wait();

  // select a processor to run the cpu task
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  rt.shutdown(e);

  int ret = rt.wait_for_shutdown();

  return ret;
}
