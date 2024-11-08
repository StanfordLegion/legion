/* Copyright 2024 Stanford University, NVIDIA Corporation
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

#ifndef CUDA_MEMCPY_H
#define CUDA_MEMCPY_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "realm/point.h"

#define CUDA_MAX_FIELD_BYTES 64
#define CUDA_MAX_BLOCKS_PER_GRID 2048

namespace Realm {
  namespace Cuda {

    template <size_t N, typename Offset_t = size_t>
    struct alignas(8) AffineSubRect {
      // Extent of the ND array
      Offset_t strides[N - 1];
      // Address of the ND array
      uintptr_t addr;
    };

    template <size_t N, typename Offset_t = size_t>
    struct alignas(AffineSubRect<N, Offset_t>) AffineCopyPair {
      AffineSubRect<N, Offset_t> src;
      AffineSubRect<N, Offset_t> dst;
      // Extent of the ND sub-rect
      Offset_t extents[N];
      // Product of the extents for fast lookup, which is the same across
      // the pair
      Offset_t volume;
    };

    template <size_t N, typename Offset_t = size_t>
    struct alignas(8) AffineFillRect {
      Offset_t strides[N - 1];
      // Product of the extents for fast lookup, which is the same across
      // the pair
      Offset_t volume;
      // Extent of the ND sub-rect
      Offset_t extents[N];
      uintptr_t addr;
    };

    static const size_t MAX_CUDA_PARAM_CONSTBANK_SIZE = 4 * 1024;

    template <size_t N, typename Offset_t = size_t,
              size_t MAX_RECTS = (MAX_CUDA_PARAM_CONSTBANK_SIZE - 20) /
                                 sizeof(AffineFillRect<N, Offset_t>)>
    struct AffineFillInfo {
      enum
      {
        MAX_NUM_RECTS = MAX_RECTS,
        DIM = N
      };
      alignas(16) unsigned char fill_value[16];
      unsigned short num_rects;
      AffineFillRect<N, Offset_t> subrects[MAX_RECTS];
    };

    // Fills the ND rectangle starting at addr with the contents of the first element
    // who's size is given by fill_elem_size. (This assumes the fill data is already
    // copied to the first element)
    template <size_t N, typename Offset_t = size_t>
    struct AffineLargeFillInfo {
      enum
      {
        DIM = N
      };
      Offset_t extents[DIM];
      Offset_t strides[DIM - 1];
      Offset_t volume;
      Offset_t fill_elem_size;
      uintptr_t addr;
    };

    template <size_t N, typename Offset_t = size_t,
              size_t MAX_RECTS = (MAX_CUDA_PARAM_CONSTBANK_SIZE - 2) /
                                 sizeof(AffineCopyPair<N, Offset_t>)>
    struct alignas(AffineCopyPair<N, Offset_t>) AffineCopyInfo {
      enum { MAX_NUM_RECTS = MAX_RECTS, DIM = N };

      AffineCopyPair<N, Offset_t> subrects[MAX_RECTS];
      unsigned short num_rects;
    };

    template <typename Offset_t>
    struct MemcpyTransposeInfo {
      Offset_t extents[3];
      Offset_t src_strides[2];
      Offset_t dst_strides[2];
      Offset_t tile_size;
      uintptr_t dst;
      uintptr_t src;
    };

    template <size_t N, typename Offset_t = size_t>
    struct MemcpyIndirectInfo {
      Offset_t volume;
      Offset_t field_size;
      Offset_t src_strides[N];
      Offset_t dst_strides[N];
      uintptr_t src_ind_addr;
      uintptr_t dst_ind_addr;
      uintptr_t src_addr;
      uintptr_t dst_addr;
    };

    static const size_t CUDA_MAX_DIM = REALM_MAX_DIM < 3 ? REALM_MAX_DIM : 3;
  } // namespace Cuda
} // namespace Realm

#endif // CUDA_MEMCPY_H


