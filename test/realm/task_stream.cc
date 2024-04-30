
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

enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  GPU_TASK,
  GPU_TASK_WITH_STREAM,
};

#ifdef REALM_USE_CUDA
void gpu_kernel_wrapper(cudaStream_t stream = 0);
#else
void gpu_kernel_wrapper(hipStream_t stream = 0);
#endif

void gpu_task(const void *args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p) 
{
  log_app.print() << "Hello from GPU task";
#ifdef REALM_USE_CUDA
  cudaStream_t task_stream = Cuda::get_task_cuda_stream();
  Cuda::set_task_ctxsync_required(false);
#else
  hipStream_t task_stream = Hip::get_task_hip_stream();
  Hip::set_task_ctxsync_required(false);
#endif
  gpu_kernel_wrapper(task_stream);
}

#ifdef REALM_USE_CUDA
void gpu_task_with_stream(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p, cudaStream_t stream) 
#else
void gpu_task_with_stream(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p, hipStream_t stream) 
#endif
{
  log_app.print() << "Hello from GPU task with stream " << stream;
#ifdef REALM_USE_CUDA
  cudaStream_t task_stream = Cuda::get_task_cuda_stream();
  Cuda::set_task_ctxsync_required(false);
#else
  hipStream_t task_stream = Hip::get_task_hip_stream();
  Hip::set_task_ctxsync_required(false);
#endif
  assert(task_stream == stream);
  gpu_kernel_wrapper(stream);
}

void top_level_task(const void *args, size_t arglen, const void *userdata,
               size_t userlen, Processor p) 
{
  Processor gpu = Machine::ProcessorQuery(Machine::get_machine())
    .only_kind(Processor::TOC_PROC).first();
  assert(gpu.exists());

  gpu.spawn(GPU_TASK, NULL, 0).wait();

  gpu.spawn(GPU_TASK_WITH_STREAM, NULL, 0).wait();
}

int main(int argc, char **argv) 
{
  Runtime rt;
  rt.init(&argc, &argv);
  
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   TOP_LEVEL_TASK,
                                   CodeDescriptor(top_level_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   GPU_TASK,
                                   CodeDescriptor(gpu_task),
                                   ProfilingRequestSet(),
                                   0, 0).wait();
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   GPU_TASK_WITH_STREAM,
                                   CodeDescriptor(gpu_task_with_stream),
                                   ProfilingRequestSet(),
                                   0, 0).wait();

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
