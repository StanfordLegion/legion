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
#ifdef LEGION_COMPLEX_LEXICOGRAPHIC_COMPARE
  /*static*/ const complex<__half> MaxReduction<complex<__half> >::identity = complex<__half>(MaxReduction<__half>::identity, MaxReduction<__half>::identity);
  /*static*/ const complex<__half> MinReduction<complex<__half> >::identity = complex<__half>(MinReduction<__half>::identity, MinReduction<__half>::identity);
#endif
#endif
  /*static*/ const complex<float> SumReduction<complex<float> >::identity = complex<float>(0.f, 0.f);
  /*static*/ const complex<float> DiffReduction<complex<float> >::identity = complex<float>(0.f, 0.f);
  /*static*/ const complex<float> ProdReduction<complex<float> >::identity = complex<float>(1.f, 0.f);
  /*static*/ const complex<float> DivReduction<complex<float> >::identity = complex<float>(1.f, 0.f);
#ifdef LEGION_COMPLEX_LEXICOGRAPHIC_COMPARE
  /*static*/ const complex<float> MaxReduction<complex<float>>::identity = complex<float>(MaxReduction<float>::identity, MaxReduction<float>::identity);
  /*static*/ const complex<float> MinReduction<complex<float>>::identity = complex<float>(MinReduction<double>::identity, MinReduction<double>::identity);
#endif

  /*static*/ const complex<double> SumReduction<complex<double> >::identity = complex<double>(0.f, 0.f);
#endif

#define REGISTER_BUILTIN_REDOP(id, type)  \
  register_reduction_op(id, Realm::ReductionOpUntyped::create_reduction_op< \
      type >(), NULL, NULL, false)

  namespace Internal {
    /*static*/ void Runtime::register_builtin_reduction_operators(void)
    {
      // Sum Reductions
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_OR_BOOL, SumReduction<bool>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_INT8, SumReduction<int8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_INT16, SumReduction<int16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_INT32, SumReduction<int32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_INT64, SumReduction<int64_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_UINT8, SumReduction<uint8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_UINT16, SumReduction<uint16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_UINT32, SumReduction<uint32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_UINT64, SumReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_FLOAT16, SumReduction<__half>);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_FLOAT32, SumReduction<float>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_FLOAT64, SumReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_COMPLEX32, SumReduction<complex<__half> >);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_COMPLEX64, SumReduction<complex<float> >);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_SUM_COMPLEX128, SumReduction<complex<double> >);
#endif
      // Difference Reductions
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_INT8, DiffReduction<int8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_INT16, DiffReduction<int16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_INT32, DiffReduction<int32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_INT64, DiffReduction<int64_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_UINT8, DiffReduction<uint8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_UINT16, DiffReduction<uint16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_UINT32, DiffReduction<uint32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_UINT64, DiffReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_FLOAT16, DiffReduction<__half>);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_FLOAT32, DiffReduction<float>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_FLOAT64, DiffReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_COMPLEX32, DiffReduction<complex<__half> >);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIFF_COMPLEX64, DiffReduction<complex<float> >);
#endif
      // Product Reductions
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_AND_BOOL, ProdReduction<bool>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_INT8, ProdReduction<int8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_INT16, ProdReduction<int16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_INT32, ProdReduction<int32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_INT64, ProdReduction<int64_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_UINT8, ProdReduction<uint8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_UINT16, ProdReduction<uint16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_UINT32, ProdReduction<uint32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_UINT64, ProdReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_FLOAT16, ProdReduction<__half>);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_FLOAT32, ProdReduction<float>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_FLOAT64, ProdReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_COMPLEX32, ProdReduction<complex<__half> >);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_PROD_COMPLEX64, ProdReduction<complex<float> >);
#endif
      // Divide Reductions
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_INT8, DivReduction<int8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_INT16, DivReduction<int16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_INT32, DivReduction<int32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_INT64, DivReduction<int64_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_UINT8, DivReduction<uint8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_UINT16, DivReduction<uint16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_UINT32, DivReduction<uint32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_UINT64, DivReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_FLOAT16, DivReduction<__half>);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_FLOAT32, DivReduction<float>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_FLOAT64, DivReduction<double>);
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_COMPLEX32, DivReduction<complex<__half> >);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_DIV_COMPLEX64, DivReduction<complex<float> >);
#endif
      // Max Reductions
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_BOOL, MaxReduction<bool>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_INT8, MaxReduction<int8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_INT16, MaxReduction<int16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_INT32, MaxReduction<int32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_INT64, MaxReduction<int64_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_UINT8, MaxReduction<uint8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_UINT16, MaxReduction<uint16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_UINT32, MaxReduction<uint32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_UINT64, MaxReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_FLOAT16, MaxReduction<__half>);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_FLOAT32, MaxReduction<float>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MAX_FLOAT64, MaxReduction<double>);
      // Min Reductions
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_BOOL, MinReduction<bool>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_INT8, MinReduction<int8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_INT16, MinReduction<int16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_INT32, MinReduction<int32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_INT64, MinReduction<int64_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_UINT8, MinReduction<uint8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_UINT16, MinReduction<uint16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_UINT32, MinReduction<uint32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_UINT64, MinReduction<uint64_t>);
#ifdef LEGION_REDOP_HALF
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_FLOAT16, MinReduction<__half>);
#endif
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_FLOAT32, MinReduction<float>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_MIN_FLOAT64, MinReduction<double>);
      // Bitwise-OR Reductions
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_OR_INT8, OrReduction<int8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_OR_INT16, OrReduction<int16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_OR_INT32, OrReduction<int32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_OR_INT64, OrReduction<int64_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_OR_UINT8, OrReduction<uint8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_OR_UINT16, OrReduction<uint16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_OR_UINT32, OrReduction<uint32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_OR_UINT64, OrReduction<uint64_t>);
      // Bitwise-AND Reductions
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_AND_INT8, AndReduction<int8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_AND_INT16, AndReduction<int16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_AND_INT32, AndReduction<int32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_AND_INT64, AndReduction<int64_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_AND_UINT8, AndReduction<uint8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_AND_UINT16, AndReduction<uint16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_AND_UINT32, AndReduction<uint32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_AND_UINT64, AndReduction<uint64_t>);
      // Bitwise-XOR Reductions
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_XOR_BOOL, XorReduction<bool>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_XOR_INT8, XorReduction<int8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_XOR_INT16, XorReduction<int16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_XOR_INT32, XorReduction<int32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_XOR_INT64, XorReduction<int64_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_XOR_UINT8, XorReduction<uint8_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_XOR_UINT16, XorReduction<uint16_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_XOR_UINT32, XorReduction<uint32_t>);
      REGISTER_BUILTIN_REDOP(LEGION_REDOP_XOR_UINT64, XorReduction<uint64_t>);
    }
  }; // namespace Internal
}; // namespace Legion

