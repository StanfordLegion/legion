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
#ifndef REALM_HIP_REDOP_H
#define REALM_HIP_REDOP_H

#include "realm/realm_config.h"
#include "hip_reduc.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#ifdef REALM_USE_HIP
#include <hip/hip_runtime.h>
#endif
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace Realm {
  namespace Hip {
    // The general formula for a linearized index is the following:
    // I = \sum_{i=0}^{N} v_i (\prod_{j=0}^{i-1} D_j)
    // This implies that the general form for a coordinate is:
    // v_i = mod(div, D_j) where div is the floored dividend of all the dimensions D
    // of the earlier dimensions, e.g. for 3D:
    // I = z * D_y * D_x + y * D_x + x
    // x = mod(I, D_x)                      = I % D_x
    // y = mod(div(I, D_x), D_y)            = (I / D_x) % D_y
    // z = mod(div(div((I,D_x),D_y), D_z)   = ((I / D_x) / D_y) % D_z
    template <typename Offset_t = size_t>
    static __device__ inline void index_to_coords(Offset_t *coords, Offset_t index,
                                                  const Offset_t *extents,
                                                  const size_t elem_size)
    {
      size_t div = index;
      const unsigned n = 3;
#pragma unroll
      for(int i = 0; i < n - 1; i++) {
        size_t div_tmp = div / extents[i];
        coords[i] = div - div_tmp * extents[i];
        div = div_tmp;
      }
      coords[n - 1] = div;
      coords[0] = coords[0] * elem_size;
    }

    template <typename Offset_t = size_t>
    static __device__ inline size_t coords_to_index(const Offset_t *coords,
                                                    const Offset_t *strides,
                                                    const size_t elem_size)
    {
      size_t i = 0;
      size_t vol = 1;
      int d = 0;
      const unsigned n = 3;
#pragma unroll
      for(; d < n - 1; d++) {
        i += vol * coords[d];
        vol *= strides[d];
      }

      i += vol * coords[d];
      i = i / elem_size;
      return i;
    }

    template <typename Offset_t = size_t>
    static __device__ inline size_t coords_to_index_transpose(const Offset_t *coords,
                                                              const Offset_t *strides)
    {
      size_t i = 0;
      i = coords[1] * strides[0] + coords[2] * strides[1] + coords[0];
      return i;
    }

    namespace ReductionKernelsAdvanced {
      template <typename REDOP, bool EXCL>
      __global__ void fold_hip_kernel(Realm::Hip::AffineReducInfo<3> info, REDOP redop)
      {
        size_t off = blockIdx.x * blockDim.x + threadIdx.x;
        Realm::Hip::AffineReducPair<3> &current_info = info.subrects[0];
        size_t vol = current_info.volume;
        size_t num_elems_rhs = current_info.src.elem_size / sizeof(typename REDOP::RHS);
        size_t redop_rhs_size = sizeof(typename REDOP::RHS);
        typename REDOP::RHS *dst =
            reinterpret_cast<typename REDOP::RHS *>(current_info.dst.addr);
        typename REDOP::RHS *src =
            reinterpret_cast<typename REDOP::RHS *>(current_info.src.addr);
        for(size_t idx = off; idx < vol; idx += blockDim.x * gridDim.x) {
          size_t coords[3];
          index_to_coords<size_t>(coords, idx, current_info.extents, redop_rhs_size);
          const size_t src_idx =
              coords_to_index<size_t>(coords, current_info.src.strides, redop_rhs_size);
          const size_t dst_idx =
              coords_to_index<size_t>(coords, current_info.dst.strides, redop_rhs_size);
          redop.template fold_hip<EXCL>(
              *reinterpret_cast<typename REDOP::RHS *>(&dst[dst_idx * num_elems_rhs]),
              *reinterpret_cast<const typename REDOP::RHS *>(
                  &src[src_idx * num_elems_rhs]));
        }
      }

      template <typename REDOP, bool EXCL>
      __global__ void apply_hip_kernel(Realm::Hip::AffineReducInfo<3> info, REDOP redop)
      {
        size_t off = blockIdx.x * blockDim.x + threadIdx.x;
        Realm::Hip::AffineReducPair<3> &current_info = info.subrects[0];
        size_t vol = current_info.volume;
        size_t num_elems_lhs = current_info.dst.elem_size / sizeof(typename REDOP::LHS);
        size_t num_elems_rhs = current_info.src.elem_size / sizeof(typename REDOP::RHS);
        size_t redop_lhs_size = sizeof(typename REDOP::LHS);
        size_t redop_rhs_size = sizeof(typename REDOP::RHS);
        typename REDOP::LHS *dst =
            reinterpret_cast<typename REDOP::LHS *>(current_info.dst.addr);
        typename REDOP::RHS *src =
            reinterpret_cast<typename REDOP::RHS *>(current_info.src.addr);
        for(size_t idx = off; idx < vol; idx += blockDim.x * gridDim.x) {
          size_t coords[3];
          index_to_coords<size_t>(coords, idx, current_info.extents, redop_rhs_size);
          const size_t src_idx =
              coords_to_index<size_t>(coords, current_info.src.strides, redop_rhs_size);
          const size_t dst_idx =
              coords_to_index<size_t>(coords, current_info.dst.strides, redop_lhs_size);
          redop.template apply_hip<EXCL>(
              *reinterpret_cast<typename REDOP::LHS *>(&(dst[dst_idx * num_elems_lhs])),
              *reinterpret_cast<const typename REDOP::RHS *>(
                  &src[src_idx * num_elems_rhs]));
        }
      }
    }; // namespace ReductionKernelsAdvanced

    namespace ReductionKernelsTranspose {
      template <typename REDOP, bool EXCL>
      __global__ void fold_hip_kernel(Realm::Hip::MemReducInfo<size_t> current_info,
                                      REDOP redop)
      {
        size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        size_t vol = current_info.volume;
        size_t num_elems = current_info.elem_size / sizeof(typename REDOP::RHS);
        typename REDOP::RHS *dst =
            reinterpret_cast<typename REDOP::RHS *>(current_info.dst);
        typename REDOP::RHS *src =
            reinterpret_cast<typename REDOP::RHS *>(current_info.src);
        for(size_t idx = offset; idx < vol; idx += blockDim.x * gridDim.x) {
          size_t coords[3];
          index_to_coords<size_t>(coords, idx, current_info.extents, 1);
          const size_t src_idx =
              coords_to_index_transpose<size_t>(coords, current_info.src_strides);
          const size_t dst_idx =
              coords_to_index_transpose<size_t>(coords, current_info.dst_strides);
          redop.template fold_hip<EXCL>(
              *reinterpret_cast<typename REDOP::RHS *>(&dst[dst_idx * num_elems]),
              *reinterpret_cast<const typename REDOP::RHS *>(&src[src_idx * num_elems]));
        }
      }

      template <typename REDOP, bool EXCL>
      __global__ void apply_hip_kernel(Realm::Hip::MemReducInfo<size_t> current_info,
                                       REDOP redop)
      {
        const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
        size_t vol = current_info.volume;
        size_t num_elems = current_info.elem_size / sizeof(typename REDOP::RHS);
        typename REDOP::LHS *dst =
            reinterpret_cast<typename REDOP::LHS *>(current_info.dst);
        typename REDOP::RHS *src =
            reinterpret_cast<typename REDOP::RHS *>(current_info.src);

        for(size_t idx = offset; idx < vol; idx += blockDim.x * gridDim.x) {
          size_t coords[3];
          index_to_coords<size_t>(coords, idx, current_info.extents, 1);
          const size_t src_idx =
              coords_to_index_transpose<size_t>(coords, current_info.src_strides);
          const size_t dst_idx =
              coords_to_index_transpose<size_t>(coords, current_info.dst_strides);
          redop.template apply_hip<EXCL>(
              *reinterpret_cast<typename REDOP::LHS *>(&dst[dst_idx * num_elems]),
              *reinterpret_cast<const typename REDOP::RHS *>(&src[src_idx * num_elems]));
        }
      }
    }; // namespace ReductionKernelsTranspose

    // the ability to add HIP kernels to a reduction op is only available
    //  when using a compiler that understands HIP
    namespace ReductionKernels {

      template <typename LHS, typename RHS, typename F>
      __device__ void iter_hip_kernel(uintptr_t lhs_base, uintptr_t lhs_stride,
                                      uintptr_t rhs_base, uintptr_t rhs_stride,
                                      size_t count, F func, void *context = nullptr)
      {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        for(size_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
          (*func)(*reinterpret_cast<LHS *>(lhs_base + idx * lhs_stride),
                  *reinterpret_cast<const RHS *>(rhs_base + idx * rhs_stride), context);
        }
      }

      template <typename REDOP, bool EXCL>
      __device__ void redop_apply_wrapper(typename REDOP::LHS &lhs,
                                          const typename REDOP::RHS &rhs, void *context)
      {
        REDOP &redop = *reinterpret_cast<REDOP *>(context);
        redop.template apply_hip<EXCL>(lhs, rhs);
      }
      template <typename REDOP, bool EXCL>
      __device__ void redop_fold_wrapper(typename REDOP::RHS &rhs1,
                                         const typename REDOP::RHS &rhs2, void *context)
      {
        REDOP &redop = *reinterpret_cast<REDOP *>(context);
        redop.template fold_hip<EXCL>(rhs1, rhs2);
      }

      template <typename REDOP, bool EXCL>
      __global__ void apply_hip_kernel(uintptr_t lhs_base, uintptr_t lhs_stride,
                                       uintptr_t rhs_base, uintptr_t rhs_stride,
                                       size_t count, REDOP redop)
      {
        iter_hip_kernel<typename REDOP::LHS, typename REDOP::RHS>(
            lhs_base, lhs_stride, rhs_base, rhs_stride, count,
            redop_apply_wrapper<REDOP, EXCL>, (void *)&redop);
      }

      template <typename REDOP, bool EXCL>
      __global__ void fold_hip_kernel(uintptr_t rhs1_base, uintptr_t rhs1_stride,
                                      uintptr_t rhs2_base, uintptr_t rhs2_stride,
                                      size_t count, REDOP redop)
      {
        iter_hip_kernel<typename REDOP::RHS, typename REDOP::RHS>(
            rhs1_base, rhs1_stride, rhs2_base, rhs2_stride, count,
            redop_fold_wrapper<REDOP, EXCL>, (void *)&redop);
      }
    }; // namespace ReductionKernels

    template <typename REDOP, typename T /*= ReductionOpUntyped*/>
    void add_hip_redop_kernels_advanced(T *redop)
    {
      redop->hip_apply_excl_fn_advanced = reinterpret_cast<void *>(
          &ReductionKernelsAdvanced::apply_hip_kernel<REDOP, true>);
      redop->hip_apply_nonexcl_fn_advanced = reinterpret_cast<void *>(
          &ReductionKernelsAdvanced::apply_hip_kernel<REDOP, false>);
      redop->hip_fold_excl_fn_advanced = reinterpret_cast<void *>(
          &ReductionKernelsAdvanced::fold_hip_kernel<REDOP, true>);
      redop->hip_fold_nonexcl_fn_advanced = reinterpret_cast<void *>(
          &ReductionKernelsAdvanced::fold_hip_kernel<REDOP, false>);

      redop->hip_apply_excl_fn_transpose = reinterpret_cast<void *>(
          &ReductionKernelsTranspose::apply_hip_kernel<REDOP, true>);
      redop->hip_apply_nonexcl_fn_transpose = reinterpret_cast<void *>(
          &ReductionKernelsTranspose::apply_hip_kernel<REDOP, false>);
      redop->hip_fold_excl_fn_transpose = reinterpret_cast<void *>(
          &ReductionKernelsTranspose::fold_hip_kernel<REDOP, true>);
      redop->hip_fold_nonexcl_fn_transpose = reinterpret_cast<void *>(
          &ReductionKernelsTranspose::fold_hip_kernel<REDOP, false>);
    }

    // this helper adds the appropriate kernels for REDOP to a ReductionOpUntyped,
    // although the latter is templated to work around circular include deps
    template <typename REDOP, typename T /*= ReductionOpUntyped*/>
    void add_hip_redop_kernels(T *redop)
    {
      // store the host proxy function pointer, as it's the same for all
      // devices - translation to actual hip functions happens later
      redop->hip_apply_excl_fn =
          reinterpret_cast<void *>(&ReductionKernels::apply_hip_kernel<REDOP, true>);
      redop->hip_apply_nonexcl_fn =
          reinterpret_cast<void *>(&ReductionKernels::apply_hip_kernel<REDOP, false>);
      redop->hip_fold_excl_fn =
          reinterpret_cast<void *>(&ReductionKernels::fold_hip_kernel<REDOP, true>);
      redop->hip_fold_nonexcl_fn =
          reinterpret_cast<void *>(&ReductionKernels::fold_hip_kernel<REDOP, false>);
      add_hip_redop_kernels_advanced<REDOP, T>(redop);
    }
  }; // namespace Hip
};   // namespace Realm
#endif
#endif
