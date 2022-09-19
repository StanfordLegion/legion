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

#ifndef REALM_CUDA_REDOP_H
#define REALM_CUDA_REDOP_H

#include "realm/realm_config.h"

namespace Realm {

  namespace Cuda {

#ifdef __CUDACC__

    typedef cudaError_t (*PFN_cudaLaunchKernel)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
#if CUDART_VERSION >= 11000
    typedef cudaError_t (*PFN_cudaGetFuncBySymbol)(cudaFunction_t* functionPtr, const void* symbolPtr);
#endif

    // the ability to add CUDA kernels to a reduction op is only available
    //  when using a compiler that understands CUDA
    namespace ReductionKernels {
      template <typename REDOP, bool EXCL>
      __global__ void apply_cuda_kernel(uintptr_t lhs_base,
                                        uintptr_t lhs_stride,
                                        uintptr_t rhs_base,
                                        uintptr_t rhs_stride,
                                        size_t count,
                                        REDOP redop)
      {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        for(size_t idx = tid; tid < count; tid += blockDim.x * gridDim.x)
          redop.template apply_cuda<EXCL>(*reinterpret_cast<typename REDOP::LHS *>(lhs_base + idx * lhs_stride),
                                          *reinterpret_cast<const typename REDOP::RHS *>(rhs_base + idx * rhs_stride));
      }

      template <typename REDOP, bool EXCL>
      __global__ void fold_cuda_kernel(uintptr_t rhs1_base,
                                       uintptr_t rhs1_stride,
                                       uintptr_t rhs2_base,
                                       uintptr_t rhs2_stride,
                                       size_t count,
                                       REDOP redop)
      {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        for(size_t idx = tid; tid < count; tid += blockDim.x * gridDim.x)
          redop.template fold_cuda<EXCL>(*reinterpret_cast<typename REDOP::RHS *>(rhs1_base + idx * rhs1_stride),
                                         *reinterpret_cast<const typename REDOP::RHS *>(rhs2_base + idx * rhs2_stride));
      }
    };

    // this helper adds the appropriate kernels for REDOP to a
    // ReductionOpUntyped,
    //  although the latter is templated to work around circular include deps
    template <typename REDOP, typename T /*= ReductionOpUntyped*/>
    void add_cuda_redop_kernels(T *redop) {
      // store the host proxy function pointer, as it's the same for all
      //  devices - translation to actual cudaFunction_t's happens later
      redop->cuda_apply_excl_fn = reinterpret_cast<void *>(
          &ReductionKernels::apply_cuda_kernel<REDOP, true>);
      redop->cuda_apply_nonexcl_fn = reinterpret_cast<void *>(
          &ReductionKernels::apply_cuda_kernel<REDOP, false>);
      redop->cuda_fold_excl_fn = reinterpret_cast<void *>(
          &ReductionKernels::fold_cuda_kernel<REDOP, true>);
      redop->cuda_fold_nonexcl_fn = reinterpret_cast<void *>(
          &ReductionKernels::fold_cuda_kernel<REDOP, false>);
      // Store some connections to the client's runtime instance that will be
      // used for launching the above instantiations
      // We use static cast here for type safety, as cudart is not ABI stable,
      // so we want to ensure the functions used here match our expectations
      PFN_cudaLaunchKernel launch_fn =
          static_cast<PFN_cudaLaunchKernel>(cudaLaunchKernel);
      redop->cudaLaunchKernel_fn = reinterpret_cast<void *>(launch_fn);
#if CUDART_VERSION >= 11000
      PFN_cudaGetFuncBySymbol symbol_fn =
          static_cast<PFN_cudaGetFuncBySymbol>(cudaGetFuncBySymbol);
      redop->cudaGetFuncBySymbol_fn = reinterpret_cast<void *>(symbol_fn);
#endif
    }
#endif

  }; // namespace Cuda

}; // namespace Realm

#endif
