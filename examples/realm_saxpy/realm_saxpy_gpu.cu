/* Copyright 2018 Stanford University, NVIDIA Corporation
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
void gpu_saxpy(const float alpha,
	       //const int num_elements,
	       Rect<1> bounds,
	       AffineAccessor<float, 1> ra_x,
	       AffineAccessor<float, 1> ra_y,
	       AffineAccessor<float, 1> ra_z)
	       
	       //               const float *x, const float *y, float *z)
{
  int p = bounds.lo + (blockIdx.x * blockDim.x) + threadIdx.x;
  if (p <= bounds.hi)
    ra_z[p] += alpha * ra_x[p] + ra_y[p];
}

__host__
void gpu_saxpy_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p)
{
  assert(arglen == sizeof(SaxpyArgs));
  const SaxpyArgs *saxpy_args = (const SaxpyArgs*)args;
  printf("Running GPU Saxpy Task\n\n");

  // get affine accessors for each of our three instances
  AffineAccessor<float, 1> ra_x = AffineAccessor<float, 1>(saxpy_args->x_inst,
							   FID_X);
  AffineAccessor<float, 1> ra_y = AffineAccessor<float, 1>(saxpy_args->y_inst,
							   FID_Y);
  AffineAccessor<float, 1> ra_z = AffineAccessor<float, 1>(saxpy_args->z_inst,
							   FID_Z);

  size_t num_elements = saxpy_args->bounds.volume();

  size_t cta_threads = 256;
  size_t total_ctas = (num_elements + (cta_threads-1))/cta_threads; 
  gpu_saxpy<<<total_ctas, cta_threads>>>(saxpy_args->alpha, saxpy_args->bounds,
					 ra_x, ra_y, ra_z);
  // LOOK: NO WAIT! :)
}

