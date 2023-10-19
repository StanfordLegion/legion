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

// The general formula for a linearized index is the following:
// I = \sum_{i=0}^{N} v_i (\prod_{j=0}^{i-1} D_j)
// This implies that the general form for a coordinate is:
// v_i = mod(div, D_j) where div is the floored dividend of all the dimensions D
// of the earlier dimensions, e.g. for 3D:
// I = z * D_y * D_x + y * D_x + x
// x = mod(I, D_x)                      = I % D_x
// y = mod(div(I, D_x), D_y)            = (I / D_x) % D_y
// z = mod(div(div((I,D_x),D_y), D_z)   = ((I / D_x) / D_y) % D_z

template <size_t N, typename Offset_t = size_t>
static __device__ inline void index_to_coords(Offset_t *coords, Offset_t index,
                                              Offset_t *extents)
{
  size_t div = index;
#pragma unroll
  for(int i = 0; i < N - 1; i++) {
    size_t div_tmp = div / extents[i];
    coords[i] = div - div_tmp * extents[i];
    div = div_tmp;
  }
  coords[N - 1] = div;
}

template <size_t N, typename Offset_t = size_t>
static __device__ inline size_t coords_to_index(Offset_t *coords, Offset_t *strides)
{
  size_t i = 0;
  size_t vol = 1;
  int d = 0;

#pragma unroll
  for(; d < N - 1; d++) {
    i += vol * coords[d];
    vol *= strides[d];
  }

  i += vol * coords[d];

  return i;
}

template <typename T, typename Offset_t = size_t>
static __device__ inline void memcpy_kernel_transpose(
    Realm::Cuda::MemcpyTransposeInfo<Offset_t> info, T* tile) {
  __restrict__ T *out_base = reinterpret_cast<T *>(info.dst);
  __restrict__ T *in_base = reinterpret_cast<T *>(info.src);
  const Offset_t tile_size = info.tile_size;
  const Offset_t tidx = threadIdx.x % tile_size;
  const Offset_t tidy = (threadIdx.x / tile_size) % tile_size;
  const Offset_t grid_dimx = ((info.extents[2] + tile_size - 1) / tile_size);
  const Offset_t grid_dimy = ((info.extents[1] + tile_size - 1) / tile_size);
  const Offset_t contig_bytes = info.extents[0];
  const Offset_t chunks = contig_bytes / sizeof(T);

  const Offset_t src_stride_x =
      ((info.src_strides[0] > info.src_strides[1]) ? info.src_strides[1]
                                                   : info.src_strides[0]) /
      contig_bytes;
  const Offset_t src_stride_y =
      ((info.src_strides[0] > info.src_strides[1]) ? info.src_strides[0]
                                                   : info.src_strides[1]) /
      contig_bytes;

  const Offset_t dst_stride_y =
      ((info.dst_strides[0] > info.dst_strides[1]) ? info.dst_strides[0]
                                                   : info.dst_strides[1]) /
      contig_bytes;

  for(Offset_t block = blockIdx.x; block < grid_dimx * grid_dimy; block += gridDim.x) {
    Offset_t block_idx = block % grid_dimx;
    Offset_t block_idy = block / grid_dimx;

    Offset_t x_base = block_idx * tile_size * chunks + tidx;
    Offset_t y_base = block_idy * tile_size + tidy;

    __syncthreads();

    for(Offset_t block_offset = 0; block_offset < chunks * tile_size;
        block_offset += tile_size) {
      if(x_base + block_offset < info.extents[2] * chunks && y_base < info.extents[1]) {
        Offset_t in_tile_idx = tidx + (tile_size + 1) * tidy * chunks;

        // The purpose of this calculation is to handle XYZ -> ZYX case
        // where contig_bytes > sizeof(T)
        Offset_t x_base_idx =
            ((x_base / chunks) * (src_stride_x * chunks) + x_base % chunks);
        tile[in_tile_idx + block_offset] =
            in_base[x_base_idx + y_base * src_stride_y * chunks + block_offset];
      }
    }

    __syncthreads();

    x_base = block_idy * tile_size * chunks + tidx;
    y_base = block_idx * tile_size + tidy;

    for(Offset_t block_offset = 0; block_offset < chunks * tile_size;
        block_offset += tile_size) {
      if(x_base + block_offset < info.extents[1] * chunks && y_base < info.extents[2]) {
        Offset_t out_tile_idx =
            (tidy + (tile_size + 1) * ((tidx + block_offset) / chunks)) * chunks +
            (tidx + block_offset) % chunks;
        out_base[x_base + dst_stride_y * y_base * chunks + block_offset] =
            tile[out_tile_idx];
      }
    }
  }
}

#define MAX_UNROLL (1)

template <typename T, size_t N, typename Offset_t = size_t>
static __device__ inline void
memcpy_affine_batch(Realm::Cuda::AffineCopyPair<N, Offset_t> *info,
                    size_t nrects, size_t start_offset = 0)
{
  Offset_t offset = blockIdx.x * blockDim.x + threadIdx.x - start_offset;
  const unsigned grid_stride = gridDim.x * blockDim.x;

  for(size_t rect = 0; rect < nrects; rect++) {
    Realm::Cuda::AffineCopyPair<N, Offset_t> &current_info = info[rect];
    const Offset_t vol = current_info.volume;
    __restrict__ T *dst = reinterpret_cast<T *>(current_info.dst.addr);
    __restrict__ T *src = reinterpret_cast<T *>(current_info.src.addr);

    while(offset < vol) {
      T tmp[MAX_UNROLL];
      unsigned i;

#pragma unroll
      for(i = 0; i < MAX_UNROLL; i++) {
        Offset_t src_coords[N];
        if((offset + i * grid_stride) >= vol) {
          break;
        }
        index_to_coords<N, Offset_t>(src_coords, offset + i * grid_stride,
                                     current_info.extents);
        const size_t src_idx =
            coords_to_index<N, Offset_t>(src_coords, current_info.src.strides);
        tmp[i] = src[src_idx];
      }
      for(unsigned j = 0; j < i; j++) {
        Offset_t dst_coords[N];

        index_to_coords<N, Offset_t>(dst_coords,
                                     (offset + j * grid_stride),
                                     current_info.extents);

        const size_t dst_idx =
            coords_to_index<N, Offset_t>(dst_coords, current_info.dst.strides);
        dst[dst_idx] = tmp[j];
      }

      offset += i * grid_stride;
    }

    // Skip this rectangle as it's covered by another thread isn't the one we're working
    // on This can split the warp, and it may not coalesce again unless we sync them
    offset -= vol;
  }
}

#define MEMCPY_TEMPLATE_INST(type, dim, offt, name)                            \
  extern "C" __global__ __launch_bounds__(256, 4) void                         \
      memcpy_affine_batch##name(Realm::Cuda::AffineCopyInfo<dim, offt> info) { \
    memcpy_affine_batch<type, dim, offt>(info.subrects, info.num_rects);       \
  }

#define FILL_TEMPLATE_INST(type, dim, offt, name)                                        \
  extern "C" __global__ void fill_affine_batch##name(                                    \
      Realm::Cuda::AffineFillInfo<dim, offt> info)                                       \
  {                                                                                      \
  }

#define FILL_LARGE_TEMPLATE_INST(type, dim, offt, name) \
  extern "C" __global__ void fill_affine_large##name(   \
      Realm::Cuda::AffineLargeFillInfo<dim, offt> info) {}

#define MEMCPY_TRANSPOSE_TEMPLATE_INST(type, offt, name)                                 \
  extern "C" __global__ __launch_bounds__(1024) void memcpy_transpose##name(             \
      Realm::Cuda::MemcpyTransposeInfo<offt> info)                                       \
  {                                                                                      \
    extern __shared__ type tile_shared_##name[];                                         \
    memcpy_kernel_transpose<type, offt>(info, tile_shared_##name);                       \
  }

#define MEMCPY_INDIRECT_TEMPLATE_INST(type, dim, offt, name)                  \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_indirect##name( \
      Realm::Cuda::MemcpyUnstructuredInfo<dim> info) {}

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
