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
#include "realm/hip/hiphijack_api.h"
#endif

#include <map>
#include "legion.h"
#include "legion_types.h"

namespace Legion {
  namespace Internal {

#ifdef LEGION_GPU_REDUCTIONS
    // GPU reductions are performed with custom kernels
#define REGISTER_GPU_REDUCTION_TASK(id, type)                               \
    {                                                                       \
      CodeDescriptor realm_descriptor(gpu_reduction_helper<type>);          \
      const TaskID task_id =                                                \
              LG_TASK_ID_AVAILABLE + gpu_reduction_tasks.size();            \
      registered_events.insert(RtEvent(Processor::register_task_by_kind(    \
              Processor::TOC_PROC, false/*global*/, task_id,                \
              realm_descriptor, no_requests, NULL, 0)));                    \
      gpu_reduction_tasks[id] = task_id;                                    \
    }

    __host__
    void register_builtin_gpu_reduction_tasks(
        GPUReductionTable &gpu_reduction_tasks, std::set<RtEvent> &registered_events)
    {
      Realm::ProfilingRequestSet no_requests;

      // Register Realm task for each kind of reduction
      LEGION_REDOP_LIST(REGISTER_GPU_REDUCTION_TASK)
    }
#else
    // GPU reductions are performed by Realm, but we need to register from
    //  here in order to have cuda implementations available


    // Legion's builtin reduction ops define apply and fold as host/device
    //  methods, whereas Realm is looking for apply_cuda/fold_cuda and a
    //  'has_cuda_reductions' flag, so add those with a templated wrapper
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

    // This is defined in runtime.h, but we can't include that here due
    //  to nvcc's inability to handle some constructs
    extern void runtime_register_reduction_op(ReductionOpID redop_id,
                                              ReductionOp *redop,
                                              SerdezInitFnptr init_fnptr,
                                              SerdezFoldFnptr fold_fnptr,
                                              bool permit_duplicates,
                                              bool has_lock = false);

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

  }; 
};
