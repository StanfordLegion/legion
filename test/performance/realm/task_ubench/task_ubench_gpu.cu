/* Copyright 2024 Stanford University
 * Copyright 2024 NVIDIA
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

using namespace Realm;

#ifdef REALM_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef REALM_USE_HIP
#include "hip_cuda_compat/hip_cuda.h"
#include "realm/hip/hiphijack_api.h"
#endif

__global__ void dummy_kernel()
{}

void dummy_gpu_task(const void *args, size_t arglen, 
		    const void *userdata, size_t userlen, Processor p)
{
  int threads = 1;
  int blocks = 1;
  dummy_kernel<<<blocks, threads
#ifdef REALM_USE_HIP
                 , 0, hipGetTaskStream()
#endif
              >>>();
}
