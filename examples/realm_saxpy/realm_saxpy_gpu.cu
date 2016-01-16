/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#include "realm_saxpy.h"

__global__
void gpu_saxpy(const float alpha, const int num_elements,
               const float *x, const float *y, float *z)
{
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if (tid >= num_elements)
    return;
  z[tid] = alpha * x[tid] + y[tid];
}

__host__
void gpu_saxpy_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(SaxpyArgs));
  const SaxpyArgs *saxpy_args = (const SaxpyArgs*)args;
  printf("Running GPU Saxpy Task\n\n");
  Rect<1> actual_bounds;
  ByteOffset offsets;
  // These are all device pointers
  const float *x_ptr = (const float*)saxpy_args->x_inst.get_accessor().
          raw_dense_ptr<1>(saxpy_args->bounds, actual_bounds, offsets);
  assert(actual_bounds == saxpy_args->bounds);
  const float *y_ptr = (const float*)saxpy_args->y_inst.get_accessor().
          raw_dense_ptr<1>(saxpy_args->bounds, actual_bounds, offsets);
  assert(actual_bounds == saxpy_args->bounds);
  float *z_ptr = (float*)saxpy_args->z_inst.get_accessor().
          raw_dense_ptr<1>(saxpy_args->bounds, actual_bounds, offsets);
  size_t num_elements = actual_bounds.volume();

  size_t cta_threads = 256;
  size_t total_ctas = (num_elements + (cta_threads-1))/cta_threads; 
  gpu_saxpy<<<total_ctas, cta_threads>>>(saxpy_args->alpha, num_elements,
                                         x_ptr, y_ptr, z_ptr);
  // LOOK: NO WAIT! :)
}

