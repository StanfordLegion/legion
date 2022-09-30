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

#include <float.h>
#include "legion/legion_config.h"
#include "legion/runtime.h"

namespace Legion {

#ifdef LEGION_REDOP_HALF
  /*static*/ const __half SumReduction<__half>::identity = __half(0, false/*raw*/);
  /*static*/ const __half DiffReduction<__half>::identity = __half(0, false/*raw*/);
  /*static*/ const __half ProdReduction<__half>::identity = __half(1, false/*raw*/);
  /*static*/ const __half DivReduction<__half>::identity = __half(1, false/*raw*/);
  /*static*/ const __half MaxReduction<__half>::identity = __half(-2e10);
  /*static*/ const __half MinReduction<__half>::identity = __half(2e10);
#endif

  /*static*/ const float SumReduction<float>::identity = 0.f;
  /*static*/ const float DiffReduction<float>::identity = 0.f;
  /*static*/ const float ProdReduction<float>::identity = 1.f;
  /*static*/ const float DivReduction<float>::identity = 1.f;
  /*static*/ const float MaxReduction<float>::identity = -FLT_MAX;
  /*static*/ const float MinReduction<float>::identity = FLT_MAX;


  /*static*/ const double SumReduction<double>::identity = 0.0;
  /*static*/ const double DiffReduction<double>::identity = 0.0;
  /*static*/ const double ProdReduction<double>::identity = 1.0;
  /*static*/ const double DivReduction<double>::identity = 1.0;
  /*static*/ const double MaxReduction<double>::identity = -DBL_MAX;
  /*static*/ const double MinReduction<double>::identity = DBL_MAX;

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  /*static*/ const complex<__half> SumReduction<complex<__half> >::identity = complex<__half>(__half(0, false/*raw*/), __half(0, false/*raw*/));
  /*static*/ const complex<__half> DiffReduction<complex<__half> >::identity = complex<__half>(__half(0, false/*raw*/), __half(0, false/*raw*/));
  /*static*/ const complex<__half> ProdReduction<complex<__half> >::identity = complex<__half>(__half(1, false/*raw*/), __half(0, false/*raw*/));
  /*static*/ const complex<__half> DivReduction<complex<__half> >::identity = complex<__half>(__half(1, false/*raw*/), __half(0, false/*raw*/));
#endif
  /*static*/ const complex<float> SumReduction<complex<float> >::identity = complex<float>(0.f, 0.f);
  /*static*/ const complex<float> DiffReduction<complex<float> >::identity = complex<float>(0.f, 0.f);
  /*static*/ const complex<float> ProdReduction<complex<float> >::identity = complex<float>(1.f, 0.f);
  /*static*/ const complex<float> DivReduction<complex<float> >::identity = complex<float>(1.f, 0.f);

  /*static*/ const complex<double> SumReduction<complex<double> >::identity = complex<double>(0.f, 0.f);
#endif

#define REGISTER_BUILTIN_REDOP(id, type)  \
  register_reduction_op(id, Realm::ReductionOpUntyped::create_reduction_op< \
      type >(), NULL, NULL, false);

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
  }; // namespace Internal
}; // namespace Legion

