/* Copyright 2020 Stanford University, NVIDIA Corporation
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

#include <map>
#include "legion.h"
#include "legion_types.h"

namespace Legion {
  namespace Internal {

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
      // Sum Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_BOOL, SumReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT8, SumReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT16, SumReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT32, SumReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_INT64, SumReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_UINT8, SumReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_UINT16, SumReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_UINT32, SumReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_UINT64, SumReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_FLOAT16, SumReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_FLOAT32, SumReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_FLOAT64, SumReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_COMPLEX32, SumReduction<complex<__half> >);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_COMPLEX64, SumReduction<complex<float> >);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_SUM_COMPLEX128, SumReduction<complex<double> >);
#endif
      // Difference Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT8, DiffReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT16, DiffReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT32, DiffReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_INT64, DiffReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_UINT8, DiffReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_UINT16, DiffReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_UINT32, DiffReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_UINT64, DiffReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_FLOAT16, DiffReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_FLOAT32, DiffReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_FLOAT64, DiffReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_COMPLEX32, DiffReduction<complex<__half> >);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIFF_COMPLEX64, DiffReduction<complex<float> >);
#endif
      // Product Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_BOOL, ProdReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT8, ProdReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT16, ProdReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT32, ProdReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_INT64, ProdReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_UINT8, ProdReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_UINT16, ProdReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_UINT32, ProdReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_UINT64, ProdReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_FLOAT16, ProdReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_FLOAT32, ProdReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_FLOAT64, ProdReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_COMPLEX32, ProdReduction<complex<__half> >);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_PROD_COMPLEX64, ProdReduction<complex<float> >);
#endif
      // Divide Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT8, DivReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT16, DivReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT32, DivReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_INT64, DivReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_UINT8, DivReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_UINT16, DivReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_UINT32, DivReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_UINT64, DivReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_FLOAT16, DivReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_FLOAT32, DivReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_FLOAT64, DivReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_COMPLEX32, DivReduction<complex<__half> >);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_DIV_COMPLEX64, DivReduction<complex<float> >);
#endif
      // Max Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_BOOL, MaxReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT8, MaxReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT16, MaxReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT32, MaxReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_INT64, MaxReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_UINT8, MaxReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_UINT16, MaxReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_UINT32, MaxReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_UINT64, MaxReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_FLOAT16, MaxReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_FLOAT32, MaxReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MAX_FLOAT64, MaxReduction<double>);
      // Min Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_BOOL, MinReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT8, MinReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT16, MinReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT32, MinReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_INT64, MinReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT8, MinReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT16, MinReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT32, MinReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_UINT64, MinReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_FLOAT16, MinReduction<__half>);
#endif
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_FLOAT32, MinReduction<float>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_MIN_FLOAT64, MinReduction<double>);
      // Bitwise-OR Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_INT8, OrReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_INT16, OrReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_INT32, OrReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_INT64, OrReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_UINT8, OrReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_UINT16, OrReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_UINT32, OrReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_OR_UINT64, OrReduction<uint64_t>);
      // Bitwise-AND Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_INT8, AndReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_INT16, AndReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_INT32, AndReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_INT64, AndReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_UINT8, AndReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_UINT16, AndReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_UINT32, AndReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_AND_UINT64, AndReduction<uint64_t>);
      // Bitwise-XOR Reductions
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_BOOL, XorReduction<bool>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_INT8, XorReduction<int8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_INT16, XorReduction<int16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_INT32, XorReduction<int32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_INT64, XorReduction<int64_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_UINT8, XorReduction<uint8_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_UINT16, XorReduction<uint16_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_UINT32, XorReduction<uint32_t>);
      REGISTER_GPU_REDUCTION_TASK(LEGION_REDOP_XOR_UINT64, XorReduction<uint64_t>);
    }

  }; 
};
