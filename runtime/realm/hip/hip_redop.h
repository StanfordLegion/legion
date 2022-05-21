/* Copyright 2021 Stanford University, NVIDIA Corporation
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

#ifdef REALM_USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace Realm {

  namespace Hip {

#if defined (__CUDACC__) || defined (__HIPCC__)
    // the ability to add CUDA kernels to a reduction op is only available
    //  when using a compiler that understands CUDA
    namespace ReductionKernels {
      template <typename REDOP, bool EXCL>
      __global__ void apply_hip_kernel(uintptr_t lhs_base,
                                        uintptr_t lhs_stride,
                                        uintptr_t rhs_base,
                                        uintptr_t rhs_stride,
                                        size_t count,
                                        REDOP redop)
      {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        for(size_t idx = tid; tid < count; tid += blockDim.x * gridDim.x)
          redop.template apply_hip<EXCL>(*reinterpret_cast<typename REDOP::LHS *>(lhs_base + idx * lhs_stride),
                                 *reinterpret_cast<const typename REDOP::RHS *>(rhs_base + idx * rhs_stride));
      }

      template <typename REDOP, bool EXCL>
      __global__ void fold_hip_kernel(uintptr_t rhs1_base,
                                       uintptr_t rhs1_stride,
                                       uintptr_t rhs2_base,
                                       uintptr_t rhs2_stride,
                                       size_t count,
                                       REDOP redop)
      {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        for(size_t idx = tid; tid < count; tid += blockDim.x * gridDim.x)
          redop.template fold_hip<EXCL>(*reinterpret_cast<typename REDOP::RHS *>(rhs1_base + idx * rhs1_stride),
                                *reinterpret_cast<const typename REDOP::RHS *>(rhs2_base + idx * rhs2_stride));
      }
    };

    // this helper adds the appropriate kernels for REDOP to a ReductionOpUntyped,
    //  although the latter is templated to work around circular include deps
    template <typename REDOP, typename T /*= ReductionOpUntyped*/>
    void add_hip_redop_kernels(T *redop)
    {
      // store the host proxy function pointer, as it's the same for all
      //  devices - translation to actual cudaFunction_t's happens later
      redop->hip_apply_excl_fn = reinterpret_cast<void *>(&ReductionKernels::apply_hip_kernel<REDOP, true>);
      redop->hip_apply_nonexcl_fn = reinterpret_cast<void *>(&ReductionKernels::apply_hip_kernel<REDOP, false>);
      redop->hip_fold_excl_fn = reinterpret_cast<void *>(&ReductionKernels::fold_hip_kernel<REDOP, true>);
      redop->hip_fold_nonexcl_fn = reinterpret_cast<void *>(&ReductionKernels::fold_hip_kernel<REDOP, false>);
    }
#endif

  }; // namespace Cuda

}; // namespace Realm

#endif
