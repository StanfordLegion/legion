/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#include <limits>

#include "legion/api/redop.h"
#include "legion/kernel/runtime.h"

namespace Legion {

#define REGISTER_BUILTIN_REDOP(id, type)                                   \
  register_reduction_op(                                                   \
      id, Realm::ReductionOpUntyped::create_reduction_op<type>(), nullptr, \
      nullptr, false);

  namespace Internal {
#if defined(LEGION_USE_CUDA)
    // Defined in legion_redop.cu
    extern void register_builtin_reduction_operators_cuda(void);
#endif
#if defined(LEGION_USE_HIP)
    // Defined in legion_redop.cpp
    extern void register_builtin_reduction_operators_hip(void);
#endif

    /*static*/ void Runtime::register_builtin_reduction_operators(void)
    {
#if defined(LEGION_USE_CUDA) || defined(LEGION_USE_HIP)
      // We need to register CUDA/HIP reductions with Realm, so that happens in
      //  legion_redop.cu/cpp
#ifdef LEGION_USE_CUDA
      register_builtin_reduction_operators_cuda();
#endif
#ifdef LEGION_USE_HIP
      register_builtin_reduction_operators_hip();
#endif
#else
      // Only CPU reductions are needed, so register them here
      LEGION_REDOP_LIST(REGISTER_BUILTIN_REDOP)
#endif
    }
  }  // namespace Internal
}  // namespace Legion
