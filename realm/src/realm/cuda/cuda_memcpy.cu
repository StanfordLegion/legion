/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
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
                                              const Offset_t *extents)
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
static __device__ inline size_t coords_to_index(Offset_t *coords, const Offset_t *strides)
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
static __device__ inline void
memcpy_kernel_transpose(Realm::Cuda::MemcpyTransposeInfo<Offset_t> info, T *tile)
{
  T* __restrict__ out_base = reinterpret_cast<T *>(info.dst);
  T* __restrict__ in_base = reinterpret_cast<T *>(info.src);
  const Offset_t tile_size = info.tile_size;

  const Offset_t tidx = threadIdx.x % tile_size;
  const Offset_t tidy = (threadIdx.x / tile_size) % tile_size;

  const Offset_t grid_dimx = ((info.extents[2] + tile_size - 1) / tile_size);
  const Offset_t grid_dimy = ((info.extents[1] + tile_size - 1) / tile_size);

  const Offset_t chunks = info.extents[0] / sizeof(T);

  const Offset_t src_size_dim0 = info.src_strides[0] / sizeof(T);
  const Offset_t src_size_dim1 = info.src_strides[1] / sizeof(T);

  const Offset_t dst_size_dim1 = info.dst_strides[1] / sizeof(T);
  const Offset_t dst_size_dim0 = info.dst_strides[0] / sizeof(T);

  for(Offset_t block = blockIdx.x; block < grid_dimx * grid_dimy; block += gridDim.x) {
    Offset_t block_idx = block % grid_dimx;
    Offset_t block_idy = block / grid_dimx;

    Offset_t gx_tile_idx = block_idx * tile_size * chunks + tidx;
    Offset_t gy_tile_idx = block_idy * tile_size + tidy;

    __syncthreads();

    for(Offset_t block_offset = 0; block_offset < chunks * tile_size;
        block_offset += tile_size) {
      if(gx_tile_idx + block_offset < info.extents[2] * chunks &&
         gy_tile_idx < info.extents[1]) {
        Offset_t in_tile_idx = tidx + (tile_size + 1) * tidy * chunks;

        Offset_t xe = gx_tile_idx + block_offset;
        Offset_t chunk_idx = xe / chunks;
        Offset_t chunk_rem = xe % chunks;

        tile[in_tile_idx + block_offset] =
            in_base[chunk_idx * src_size_dim1 + gy_tile_idx * src_size_dim0 + chunk_rem];
      }
    }

    __syncthreads();

    gx_tile_idx = block_idy * tile_size * chunks + tidx;
    gy_tile_idx = block_idx * tile_size + tidy;

    for(Offset_t block_offset = 0; block_offset < chunks * tile_size;
        block_offset += tile_size) {
      if(gx_tile_idx + block_offset < info.extents[1] * chunks &&
         gy_tile_idx < info.extents[2]) {
        Offset_t out_tile_idx =
            (tidy + (tile_size + 1) * ((tidx + block_offset) / chunks)) * chunks +
            (tidx + block_offset) % chunks;

        Offset_t xe = gx_tile_idx + block_offset;
        Offset_t chunk_idx = xe / chunks;
        Offset_t chunk_rem = xe % chunks;

        out_base[chunk_idx * dst_size_dim0 + gy_tile_idx * dst_size_dim1 + chunk_rem] =
            tile[out_tile_idx];
      }
    }
  }
}

#define MAX_UNROLL (1)

template <typename T, size_t N, typename Offset_t = size_t>
static __device__ inline void
memcpy_multi_affine_batch(Realm::Cuda::AffineCopyPair<N, Offset_t> *info, size_t nrects,
                          size_t start_offset = 0)
{
  const Offset_t blk_stride = blockDim.x;
  const Offset_t tid_global = threadIdx.x;

  /* -------- iterate over copy rectangles -------- */
  for(size_t r = 0; r < nrects; ++r) {
    auto &cp = info[r];
    Offset_t v = cp.volume; // elements in one field
    Offset_t n = max(size_t(1), max(cp.src.num_fields, cp.dst.num_fields));

    const T *__restrict__ src = reinterpret_cast<const T *>(cp.src.addr);
    T *__restrict__ dst = reinterpret_cast<T *>(cp.dst.addr);

    /* -------- iterate over fields handled by this block -------- */
    for(Offset_t f = blockIdx.x; f < n; f += gridDim.x) {

      Offset_t src_field_base = cp.src.num_fields > 0
                                    ? cp.src.fields[f] * cp.src.field_stride
                                    : f * cp.src.field_stride;

      Offset_t dst_field_base = cp.dst.num_fields > 0
                                    ? cp.dst.fields[f] * cp.dst.field_stride
                                    : f * cp.dst.field_stride;

      Offset_t off = tid_global;

      while(off < v) {
        /* -------- issue up to MAX_UNROLL loads ---------- */
        T buf[MAX_UNROLL];
        unsigned loaded = 0;

#pragma unroll
        for(unsigned i = 0; i < MAX_UNROLL; ++i) {
          Offset_t idx = off + i * blk_stride;
          if(idx >= v) {
            break;
          }

          Offset_t src_coords[N];
          index_to_coords<N>(src_coords, idx, cp.extents);
          Offset_t src_lin = coords_to_index<N>(src_coords, cp.src.strides);
          buf[i] = src[src_field_base + src_lin];
          ++loaded;
        }

/* -------- corresponding stores ------------------ */
#pragma unroll
        for(unsigned i = 0; i < loaded; ++i) {
          Offset_t idx = off + i * blk_stride;
          Offset_t dst_coords[N];
          index_to_coords<N>(dst_coords, idx, cp.extents);
          Offset_t dst_lin = coords_to_index<N>(dst_coords, cp.dst.strides);
          dst[dst_field_base + dst_lin] = buf[i];
        }

        off += loaded * blk_stride;
      } // while off < v
    } // for each field handled by this block
  } // for each rect
}

template <typename T, size_t N, typename Offset_t = size_t>
static __device__ inline void
memcpy_affine_batch(Realm::Cuda::AffineCopyPair<N, Offset_t> *info, size_t nrects,
                    size_t start_offset = 0)
{
  Offset_t offset = blockIdx.x * blockDim.x + threadIdx.x - start_offset;
  const unsigned grid_stride = gridDim.x * blockDim.x;

  for(size_t rect = 0; rect < nrects; rect++) {
    Realm::Cuda::AffineCopyPair<N, Offset_t> &current_info = info[rect];
    const Offset_t vol = current_info.volume;
    T* __restrict__ dst = reinterpret_cast<T *>(current_info.dst.addr);
    T* __restrict__ src = reinterpret_cast<T *>(current_info.src.addr);

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

        index_to_coords<N, Offset_t>(dst_coords, (offset + j * grid_stride),
                                     current_info.extents);

        const size_t dst_idx =
            coords_to_index<N, Offset_t>(dst_coords, current_info.dst.strides);
        dst[dst_idx] = tmp[j];
      }

      offset += i * grid_stride;
    }

    // Skip this rectangle as it's covered by another thread
    // This can split the warp, and it may not coalesce again unless we sync them
    offset -= vol;
  }
}

/*
 * Scatter/gather points using indirection from/to dense buffer.
 * General assumptions:
 * 1. The src_ind/dst_ind buffer is dense and always has the same size
 * as the src/dst buffer (depending whether we are doing scatter or gather).
 * 2. src_ind_/dst_ind are accessed in a linear fashion with the base
 * type Point<N, Offset_t> per indirection element.
 * 3. src_ind/dst_ind do not have to be sorted but it should be
 * considered for coalesced access.
 *
 * TODO(apryakhin@): Consider handling ranges where src_ind/dst_ind
 * contain Rect<N, Offset_t> instead of Point<N Offset_t>.
 *
 * */

template <int N, typename T, typename DT, typename Offset_t = size_t>
static __device__ inline void
memcpy_indirect_points(Realm::Cuda::MemcpyIndirectInfo<3, Offset_t> info)
{
  Offset_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  T* __restrict__ dst_ind_base = reinterpret_cast<T *>(info.dst_ind_addr);
  T* __restrict__ src_ind_base = reinterpret_cast<T *>(info.src_ind_addr);

  Offset_t chunks = info.field_size / sizeof(DT);

  for(; offset < info.volume; offset += blockDim.x * gridDim.x) {
    Offset_t src_index = offset;
    if(info.src_ind_addr != 0) {
      Offset_t index = 0;
#pragma unroll
      for(int i = 0; i < N; i++) {
        index += src_ind_base[offset * N + i] * info.src_strides[i];
      }
      src_index = index;
    }

    Offset_t dst_index = offset;
    if(info.dst_ind_addr != 0) {
      Offset_t index = 0;
#pragma unroll
      for(int i = 0; i < N; i++) {
        index += dst_ind_base[offset * N + i] * info.dst_strides[i];
      }

      dst_index = index;
    }

    DT* __restrict__ dst =
        reinterpret_cast<DT *>(info.dst_addr + dst_index * info.field_size);
    DT* __restrict__ src =
        reinterpret_cast<DT *>(info.src_addr + src_index * info.field_size);
    for(Offset_t chunk_idx = 0; chunk_idx < chunks; chunk_idx++) {
      dst[chunk_idx] = src[chunk_idx];
    }
  }
}

template <int N, typename T, typename Offset_t = size_t>
static __device__ inline void
memfill_affine_batch(const Realm::Cuda::AffineFillInfo<N, Offset_t> &info)
{
  Offset_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned grid_stride = gridDim.x * blockDim.x;
  T fill_value = *reinterpret_cast<const T *>(info.fill_value);
  for(size_t rect = 0; rect < info.num_rects; rect++) {
    const Realm::Cuda::AffineFillRect<N, Offset_t> &current_info = info.subrects[rect];
    const Offset_t vol = current_info.volume;
    T* __restrict__ addr = reinterpret_cast<T *>(current_info.addr);
    while(offset < vol) {
      unsigned i = 0;
#pragma unroll
      for(i = 0; i < MAX_UNROLL; i++) {
        Offset_t coords[N];
        if((offset + i * grid_stride) >= vol) {
          break;
        }
        index_to_coords<N, Offset_t>(coords, offset + i * grid_stride,
                                     current_info.extents);
        const size_t idx = coords_to_index<N, Offset_t>(coords, current_info.strides);
        addr[idx] = fill_value;
      }
      offset += i * grid_stride;
    }
    // Skip this rectangle as it's covered by another thread
    // This can split the warp, and it may not coalesce again unless we sync them
    offset -= vol;
  }
}

#define MEMCPY_MULTI_TEMPLATE_INST(type, dim, offt, name)                                \
  extern "C" __global__ __launch_bounds__(256, 4) void multi_affine_batch##name(         \
      Realm::Cuda::AffineCopyInfo<dim, offt> info)                                       \
  {                                                                                      \
    memcpy_multi_affine_batch<type, dim, offt>(info.subrects, info.num_rects);           \
  }

#define MEMCPY_TEMPLATE_INST(type, dim, offt, name)                                      \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_affine_batch##name(        \
      Realm::Cuda::AffineCopyInfo<dim, offt> info)                                       \
  {                                                                                      \
    memcpy_affine_batch<type, dim, offt>(info.subrects, info.num_rects);                 \
  }

#define FILL_TEMPLATE_INST(type, dim, offt, name)                                        \
  extern "C" __global__ void fill_affine_batch##name(                                    \
      Realm::Cuda::AffineFillInfo<dim, offt> info)                                       \
  {                                                                                      \
    memfill_affine_batch<dim, type, offt>(info);                                         \
  }

#define FILL_LARGE_TEMPLATE_INST(type, dim, offt, name)                                  \
  extern "C" __global__ void fill_affine_large##name(                                    \
      Realm::Cuda::AffineLargeFillInfo<dim, offt> info)                                  \
  {}

#define MEMCPY_TRANSPOSE_TEMPLATE_INST(type, offt, name)                                 \
  extern "C" __global__ __launch_bounds__(1024) void memcpy_transpose##name(             \
      Realm::Cuda::MemcpyTransposeInfo<offt> info)                                       \
  {                                                                                      \
    extern __shared__ type tile_shared_##name[];                                         \
    memcpy_kernel_transpose<type, offt>(info, tile_shared_##name);                       \
  }

#define MEMCPY_INDIRECT_TEMPLATE_INST(addr_type, data_type, dim, offt, name)             \
  extern "C" __global__ __launch_bounds__(256, 4) void memcpy_indirect##name(            \
      Realm::Cuda::MemcpyIndirectInfo<3, offt> info)                                     \
  {                                                                                      \
    memcpy_indirect_points<dim, addr_type, data_type, offt>(info);                       \
  }

#define INST_TEMPLATES(type, sz, dim, off)                                               \
  MEMCPY_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                      \
  MEMCPY_MULTI_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                \
  FILL_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                        \
  FILL_LARGE_TEMPLATE_INST(type, dim, off, dim##D_##sz)                                  \
  MEMCPY_INDIRECT_TEMPLATE_INST(int, type, dim, off, dim##D_##sz##32)                    \
  MEMCPY_INDIRECT_TEMPLATE_INST(long long, type, dim, off, dim##D_##sz##64)

#define INST_TEMPLATES_FOR_TYPES(dim, off)                                               \
  INST_TEMPLATES(unsigned char, 8, dim, off)                                             \
  INST_TEMPLATES(unsigned short, 16, dim, off)                                           \
  INST_TEMPLATES(unsigned int, 32, dim, off)                                             \
  INST_TEMPLATES(unsigned long long, 64, dim, off)                                       \
  INST_TEMPLATES(uint4, 128, dim, off)

#define INST_TEMPLATES_FOR_DIMS()                                                        \
  INST_TEMPLATES_FOR_TYPES(1, size_t)                                                    \
  INST_TEMPLATES_FOR_TYPES(2, size_t)                                                    \
  INST_TEMPLATES_FOR_TYPES(3, size_t)

INST_TEMPLATES_FOR_DIMS()

MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned char, size_t, 8)
MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned short, size_t, 16)
MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned int, size_t, 32)
MEMCPY_TRANSPOSE_TEMPLATE_INST(unsigned long long, size_t, 64)
MEMCPY_TRANSPOSE_TEMPLATE_INST(uint4, size_t, 128)
