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

#include"realm_defines.h"

#ifdef REALM_USE_HIP
#include "hip_cuda_compat/hip_cuda.h"
#include "realm/hip/hiphijack_api.h"
#endif

#include <map>
#include "legion.h"
#include "legion_types.h"

namespace Legion {
  namespace Internal {

    // GPU reductions are performed by Realm, but we need to register from
    //  here in order to have cuda implementations available


    // Legion's builtin reduction ops define apply and fold as host/device
    //  methods, whereas Realm is looking for apply_cuda/fold_cuda and a
    //  'has_cuda/hip_reductions' flag, so add those with a templated wrapper
#ifdef LEGION_USE_CUDA
    template <typename T>
    class AddCudaReductions : public T {
    public:
      static const bool has_cuda_reductions = true;

      template <bool EXCLUSIVE>
      __device__ static void apply_cuda(typename T::LHS& lhs, typename T::RHS rhs)
      {
        T::template apply<EXCLUSIVE>(lhs, rhs);
      }

      template <bool EXCLUSIVE>
      __device__ static void fold_cuda(typename T::LHS& lhs, typename T::RHS rhs)
      {
        T::template fold<EXCLUSIVE>(lhs, rhs);
      }
    };
#endif

    // We have added the hip code here becasue we rely on the hip tool to
    // auto-generate the legion_redop.cpp from the legion_redop.cu
#ifdef LEGION_USE_HIP
    template <typename T>
    class AddHipReductions : public T {
    public:
      static const bool has_hip_reductions = true;

      template <bool EXCLUSIVE>
      __device__ static void apply_hip(typename T::LHS& lhs, typename T::RHS rhs)
      {
        T::template apply<EXCLUSIVE>(lhs, rhs);
      }

      template <bool EXCLUSIVE>
      __device__ static void fold_hip(typename T::LHS& lhs, typename T::RHS rhs)
      {
        T::template fold<EXCLUSIVE>(lhs, rhs);
      }
    };
#endif

    // This is defined in runtime.h, but we can't include that here due
    //  to nvcc's inability to handle some constructs
    extern void runtime_register_reduction_op(ReductionOpID redop_id,
                                              ReductionOp *redop,
                                              SerdezInitFnptr init_fnptr,
                                              SerdezFoldFnptr fold_fnptr,
                                              bool permit_duplicates,
                                              bool has_lock = false);

#ifdef LEGION_USE_CUDA
#define REGISTER_BUILTIN_REDOP_CUDA(id, type)                           \
  runtime_register_reduction_op(id, \
      Realm::ReductionOpUntyped::create_reduction_op< AddCudaReductions<type> \
      >(), NULL, NULL, false);

    void register_builtin_reduction_operators_cuda(void)
    {
      // Register all of our reductions
      LEGION_REDOP_LIST(REGISTER_BUILTIN_REDOP_CUDA)
    }
#endif
    
#ifdef LEGION_USE_HIP
#define REGISTER_BUILTIN_REDOP_HIP(id, type)                           \
  runtime_register_reduction_op(id, \
      Realm::ReductionOpUntyped::create_reduction_op< AddHipReductions<type> \
      >(), NULL, NULL, false);

    void register_builtin_reduction_operators_hip(void)
    {
      // Register all of our reductions
      LEGION_REDOP_LIST(REGISTER_BUILTIN_REDOP_HIP)
    }
#endif

  }; 
};
