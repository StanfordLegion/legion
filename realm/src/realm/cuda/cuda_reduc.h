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

#ifndef CUDA_REDUC_H
#define CUDA_REDUC_H

#include <cstddef>
#include <cstdint>

namespace Realm {
  namespace Cuda {
    template <typename Offset_t>
    struct MemReducInfo {
      Offset_t extents[3];
      Offset_t src_strides[2];
      Offset_t dst_strides[2];
      uintptr_t dst;
      uintptr_t src;
      size_t volume;
      size_t elem_size;
    };

    template <size_t N>
    struct AffineReducSubRect {
      // Extent of the ND array
      size_t strides[N - 1];
      size_t elem_size;
      // Address of the ND array
      uintptr_t addr;
    };

    template <size_t N>
    struct AffineReducPair {
      AffineReducSubRect<N> src;
      AffineReducSubRect<N> dst;
      // Extent of the ND sub-rect
      size_t extents[N];
      // Product of the extents for fast lookup, which is the same across
      // the pair
      size_t volume;
    };
    // TODO: batch mode, MAX_RECTS > 1
    template <size_t N, size_t MAX_RECTS = 1>
    struct AffineReducInfo {
      enum
      {
        MAX_NUM_RECTS = MAX_RECTS,
        DIM = N
      };
      AffineReducPair<N> subrects[MAX_RECTS];
      unsigned short num_rects;
    };
  } // namespace Cuda
} // namespace Realm
#endif
