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

#include "cuda_runtime.h"

extern "C" {

extern void** __cudaRegisterFatBinary(const void *);

extern void __cudaRegisterFunction(void**, const void*, char *, const char *,
    int, uint3 *, uint3 *, dim3 *, dim3 *, int *);

extern void __cudaRegisterFatBinaryEnd(void **);

void**
hijackCudaRegisterFatBinary(const void* fat_bin)
{
  return __cudaRegisterFatBinary(fat_bin);
}

void
hijackCudaRegisterFunction(void** handle, const void* host_fun,
                           char* device_fun)
{
  return __cudaRegisterFunction(handle, host_fun, device_fun,
      device_fun, -1, 0, 0, 0, 0, 0);
}

void hijackCudaRegisterFatBinaryEnd(void** handle)
{
  __cudaRegisterFatBinaryEnd(handle);
}

}

