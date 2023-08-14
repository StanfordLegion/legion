/* Copyright 2022 Stanford University, NVIDIA Corporation
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

#include <assert.h>
#include <stdio.h>

#include <iostream>

#include "realm/cuda/cuda_memcpy.h"
#include "realm/point.h"

#define MEMCPY_TEMPLATE_INST(type, dim, offt, name)                                      \
  extern "C" __global__ __launch_bounds__(256, 8) void memcpy_affine_batch##name(        \
      Realm::Cuda::AffineCopyInfo<dim, offt> info)                                       \
  {                                                                                      \
  }

#define FILL_TEMPLATE_INST(type, dim, offt, name)                                        \
  extern "C" __global__ void fill_affine_batch##name(                                    \
      Realm::Cuda::AffineFillInfo<dim, offt> info)                                       \
  {                                                                                      \
  }

#define FILL_LARGE_TEMPLATE_INST(type, dim, offt, name)                                  \
  extern "C" __global__ void fill_affine_large##name(                                    \
      Realm::Cuda::AffineLargeFillInfo<dim, offt> info)                                  \
  {                                                                                      \
  }

#define MEMCPY_TRANSPOSE_TEMPLATE_INST(type, offt, name)                                 \
  extern "C" __global__ __launch_bounds__(1024) void run_memcpy_transpose##name(         \
      Realm::Cuda::MemcpyTransposeInfo<offt> info)                                       \
  {                                                                                      \
  }

#define MEMCPY_INDIRECT_TEMPLATE_INST(type, dim, offt, name)                   \
  extern "C" __global__ __launch_bounds__(256, 8) void                         \
      run_memcpy_indirect##name(Realm::Cuda::MemcpyUnstructuredInfo<dim> info) \
  {                                                                            \
  }

#define INST_TEMPLATES(type, sz, dim, off)                                     \
  MEMCPY_TEMPLATE_INST(type, dim, off, dim##D_##sz)                            \
  FILL_TEMPLATE_INST(type, dim, off, dim##D_##sz)                              \
  FILL_LARGE_TEMPLATE_INST(type, dim, off, dim##D_##sz)                        \
  MEMCPY_INDIRECT_TEMPLATE_INST(type, dim, off, dim##D_##sz)

#define INST_TEMPLATES_FOR_TYPES(dim, off)                                     \
  INST_TEMPLATES(unsigned char, 8, dim, off)                                   \
  INST_TEMPLATES(unsigned short, 16, dim, off)                                 \
  INST_TEMPLATES(unsigned int, 32, dim, off)                                   \
  INST_TEMPLATES(unsigned long long, 64, dim, off)                             \
  INST_TEMPLATES(uint4, 128, dim, off)

#define INST_TEMPLATES_FOR_DIMS()                                              \
  INST_TEMPLATES_FOR_TYPES(1, size_t)                                          \
  INST_TEMPLATES_FOR_TYPES(2, size_t)                                          \
  INST_TEMPLATES_FOR_TYPES(3, size_t)

INST_TEMPLATES_FOR_DIMS()

MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned char, size_t, 8)
MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned short, size_t, 16)
MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned int, size_t, 32)
MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned long long, size_t, 64)
MEMCPY_TRANSPOSE_TEMPLATE_INST(uint4, size_t, 128)
