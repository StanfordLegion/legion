/* Copyright 2024 Stanford University
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
#include "realm/cuda/cuda_module.h"
#include "realm/hip/hip_module.h"

bool
regent_get_task_cuda_stream(void *stream)
{
  Realm::Cuda::CudaModule *module = Realm::Runtime::get_module<Realm::Cuda::CudaModule>();
  if (!module) {
    return false;
  }

  *static_cast<CUstream_st **>(stream) = module->get_task_cuda_stream();
  module->set_task_ctxsync_required(false);
  return true;
}

void
regent_get_task_hip_stream(void *stream)
{
  Realm::Hip::HipModule *module = Realm::Runtime::get_module<Realm::Hip::HipModule>();
  if (!module) {
    return false;
  }

  *static_cast<unifiedHipStream_t **>(stream) = module->get_task_hip_stream();
  module->set_task_ctxsync_required(false);
  return true;
}

