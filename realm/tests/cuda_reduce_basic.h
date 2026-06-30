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

#ifndef CUDA_REDUCE_BASIC_H
#define CUDA_REDUCE_BASIC_H

#include <cstdint>

// Same-type integer add reduction: LHS == RHS == uint32_t.
// sizeof_lhs == sizeof_rhs satisfies the fast-path condition in progress_xd,
// so the has_fast_kernels guard (the change under test) is actually reached.
class ReductionOpAdd {
public:
  typedef uint32_t LHS;
  typedef uint32_t RHS;

  static const RHS identity = 0;

  template <bool EXCL>
  REALM_CUDA_HD void apply(LHS &lhs, const RHS &rhs) const
  {
    if(EXCL) {
      lhs += rhs;
    } else {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      atomicAdd(&lhs, rhs);
#else
      __atomic_add_fetch(&lhs, rhs, __ATOMIC_SEQ_CST);
#endif
    }
  }

  template <bool EXCL>
  REALM_CUDA_HD void fold(RHS &rhs1, const RHS &rhs2) const
  {
    if(EXCL) {
      rhs1 += rhs2;
    } else {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      atomicAdd(&rhs1, rhs2);
#else
      __atomic_add_fetch(&rhs1, rhs2, __ATOMIC_SEQ_CST);
#endif
    }
  }

#if defined(REALM_USE_CUDA) && defined(__CUDACC__)
  static const bool has_cuda_reductions = false; // registered manually below

  template <bool EXCL>
  __device__ void apply_cuda(LHS &lhs, const RHS &rhs) const
  {
    apply<EXCL>(lhs, rhs);
  }

  template <bool EXCL>
  __device__ void fold_cuda(RHS &rhs1, const RHS &rhs2) const
  {
    fold<EXCL>(rhs1, rhs2);
  }
#endif
};

#endif // CUDA_REDUCE_BASIC_H
