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

#ifndef REALM_CUDA_REDOP_H
#define REALM_CUDA_REDOP_H

#include "realm/realm_config.h"
#include "cuda_reduc.h"
#include <stddef.h>
#include <stdint.h>

namespace Realm {

  namespace Cuda {

#ifdef __CUDACC__
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
      __global__ void fold_cuda_kernel(Realm::Cuda::AffineReducInfo<3> info, REDOP redop)
      {
        size_t off = blockIdx.x * blockDim.x + threadIdx.x;
        Realm::Cuda::AffineReducPair<3> &current_info = info.subrects[0];
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
          redop.template fold_cuda<EXCL>(
              *reinterpret_cast<typename REDOP::RHS *>(&dst[dst_idx * num_elems_rhs]),
              *reinterpret_cast<const typename REDOP::RHS *>(
                  &src[src_idx * num_elems_rhs]));
        }
      }

      template <typename REDOP, bool EXCL>
      __global__ void apply_cuda_kernel(Realm::Cuda::AffineReducInfo<3> info, REDOP redop)
      {
        size_t off = blockIdx.x * blockDim.x + threadIdx.x;
        Realm::Cuda::AffineReducPair<3> &current_info = info.subrects[0];
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
          redop.template apply_cuda<EXCL>(
              *reinterpret_cast<typename REDOP::LHS *>(&(dst[dst_idx * num_elems_lhs])),
              *reinterpret_cast<const typename REDOP::RHS *>(
                  &src[src_idx * num_elems_rhs]));
        }
      }
    }; // namespace ReductionKernelsAdvanced

    namespace ReductionKernelsTranspose {
      template <typename REDOP, bool EXCL>
      __global__ void fold_cuda_kernel(Realm::Cuda::MemReducInfo<size_t> current_info,
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
          redop.template fold_cuda<EXCL>(
              *reinterpret_cast<typename REDOP::RHS *>(&dst[dst_idx * num_elems]),
              *reinterpret_cast<const typename REDOP::RHS *>(&src[src_idx * num_elems]));
        }
      }

      template <typename REDOP, bool EXCL>
      __global__ void apply_cuda_kernel(Realm::Cuda::MemReducInfo<size_t> current_info,
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
          redop.template apply_cuda<EXCL>(
              *reinterpret_cast<typename REDOP::LHS *>(&dst[dst_idx * num_elems]),
              *reinterpret_cast<const typename REDOP::RHS *>(&src[src_idx * num_elems]));
        }
      }
    }; // namespace ReductionKernelsTranspose

    // the ability to add CUDA kernels to a reduction op is only available
    //  when using a compiler that understands CUDA
    namespace ReductionKernels {

      template <typename LHS, typename RHS, typename F>
      __device__ void iter_cuda_kernel(uintptr_t lhs_base, uintptr_t lhs_stride,
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
        redop.template apply_cuda<EXCL>(lhs, rhs);
      }
      template <typename REDOP, bool EXCL>
      __device__ void redop_fold_wrapper(typename REDOP::RHS &rhs1,
                                         const typename REDOP::RHS &rhs2, void *context)
      {
        REDOP &redop = *reinterpret_cast<REDOP *>(context);
        redop.template fold_cuda<EXCL>(rhs1, rhs2);
      }

      template <typename REDOP, bool EXCL>
      __global__ void apply_cuda_kernel(uintptr_t lhs_base, uintptr_t lhs_stride,
                                        uintptr_t rhs_base, uintptr_t rhs_stride,
                                        size_t count, REDOP redop)
      {
        iter_cuda_kernel<typename REDOP::LHS, typename REDOP::RHS>(
            lhs_base, lhs_stride, rhs_base, rhs_stride, count,
            redop_apply_wrapper<REDOP, EXCL>, (void *)&redop);
      }

      template <typename REDOP, bool EXCL>
      __global__ void fold_cuda_kernel(uintptr_t rhs1_base, uintptr_t rhs1_stride,
                                       uintptr_t rhs2_base, uintptr_t rhs2_stride,
                                       size_t count, REDOP redop)
      {
        iter_cuda_kernel<typename REDOP::RHS, typename REDOP::RHS>(
            rhs1_base, rhs1_stride, rhs2_base, rhs2_stride, count,
            redop_fold_wrapper<REDOP, EXCL>, (void *)&redop);
      }
    }; // namespace ReductionKernels

    template <typename REDOP, typename T /*= ReductionOpUntyped*/>
    void add_cuda_redop_kernels_advanced(T *redop)
    {
      redop->cuda_apply_excl_fn_advanced = reinterpret_cast<void *>(
          &ReductionKernelsAdvanced::apply_cuda_kernel<REDOP, true>);
      redop->cuda_apply_nonexcl_fn_advanced = reinterpret_cast<void *>(
          &ReductionKernelsAdvanced::apply_cuda_kernel<REDOP, false>);
      redop->cuda_fold_excl_fn_advanced = reinterpret_cast<void *>(
          &ReductionKernelsAdvanced::fold_cuda_kernel<REDOP, true>);
      redop->cuda_fold_nonexcl_fn_advanced = reinterpret_cast<void *>(
          &ReductionKernelsAdvanced::fold_cuda_kernel<REDOP, false>);

      redop->cuda_apply_excl_fn_transpose = reinterpret_cast<void *>(
          &ReductionKernelsTranspose::apply_cuda_kernel<REDOP, true>);

      redop->cuda_apply_nonexcl_fn_transpose = reinterpret_cast<void *>(
          &ReductionKernelsTranspose::apply_cuda_kernel<REDOP, false>);

      redop->cuda_fold_excl_fn_transpose = reinterpret_cast<void *>(
          &ReductionKernelsTranspose::fold_cuda_kernel<REDOP, true>);

      redop->cuda_fold_nonexcl_fn_transpose = reinterpret_cast<void *>(
          &ReductionKernelsTranspose::fold_cuda_kernel<REDOP, false>);
    }

    // this helper adds the appropriate kernels for REDOP to a
    // ReductionOpUntyped,
    //  although the latter is templated to work around circular include deps
    template <typename REDOP, typename T /*= ReductionOpUntyped*/>
    void add_cuda_redop_kernels(T *redop)
    {
      // store the host proxy function pointer, as it's the same for all
      //  devices - translation to actual cudaFunction_t's happens later
      redop->cuda_apply_excl_fn =
          reinterpret_cast<void *>(&ReductionKernels::apply_cuda_kernel<REDOP, true>);
      redop->cuda_apply_nonexcl_fn =
          reinterpret_cast<void *>(&ReductionKernels::apply_cuda_kernel<REDOP, false>);
      redop->cuda_fold_excl_fn =
          reinterpret_cast<void *>(&ReductionKernels::fold_cuda_kernel<REDOP, true>);
      redop->cuda_fold_nonexcl_fn =
          reinterpret_cast<void *>(&ReductionKernels::fold_cuda_kernel<REDOP, false>);
      add_cuda_redop_kernels_advanced<REDOP, T>(redop);
      // Store some connections to the client's runtime instance that will be
      // used for launching the above instantiations
      // We use static cast here for type safety, as cudart is not ABI stable,
      // so we want to ensure the functions used here match our expectations
      typedef cudaError_t (*PFN_cudaLaunchKernel)(const void *func, dim3 gridDim,
                                                  dim3 blockDim, void **args,
                                                  size_t sharedMem, cudaStream_t stream);
      PFN_cudaLaunchKernel launch_fn =
          static_cast<PFN_cudaLaunchKernel>(cudaLaunchKernel);
      redop->cudaLaunchKernel_fn = reinterpret_cast<void *>(launch_fn);
#if CUDART_VERSION >= 11000
      typedef cudaError_t (*PFN_cudaGetFuncBySymbol)(cudaFunction_t * functionPtr,
                                                     const void *symbolPtr);
      PFN_cudaGetFuncBySymbol symbol_fn =
          static_cast<PFN_cudaGetFuncBySymbol>(cudaGetFuncBySymbol);
      redop->cudaGetFuncBySymbol_fn = reinterpret_cast<void *>(symbol_fn);
#endif
    }
#endif

  }; // namespace Cuda

}; // namespace Realm

#endif
