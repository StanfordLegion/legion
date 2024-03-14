//------------------------------------------------------------------------------
// Copyright 2024 NVIDIA Corp.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

#include <realm.h>

using namespace Realm;

#ifdef REALM_USE_HIP
#include "hip_cuda_compat/hip_cuda.h"
#endif

__global__ void copyKernel(Rect<2> bounds, AffineAccessor<float, 2> linear_accessor,
                           cudaSurfaceObject_t surface)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;

  // Have each thread copy the value from the linear array to the surface
  for(; tid < bounds.volume(); tid += stride) {
    size_t y = tid / (bounds.hi[0] + 1);
    size_t x = tid - y * (bounds.hi[0] + 1);
    float value = linear_accessor[Point<2>(x, y)];
    surf2Dwrite<float>(value, surface, x * sizeof(float), y, cudaBoundaryModeTrap);
  }
}