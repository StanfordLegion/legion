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

#ifndef __LEGION_REDOP_H__
#define __LEGION_REDOP_H__

#include <limits.h>

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
#define COMPLEX_HALF 
#endif
#include "mathtypes/complex.h"
#endif

#ifdef LEGION_REDOP_HALF
#include "mathtypes/half.h"
#endif

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

namespace Legion {

  template<typename T>
  class SumReduction {
    // Empty definition
    // Specializations provided for each type
  };

  template<>
  class SumReduction<bool> {
  public:
    typedef bool LHS;
    typedef bool RHS;

    static const bool identity = false;
    static const int REDOP_ID = LEGION_REDOP_OR_BOOL;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class SumReduction<int8_t> {
  public:
    typedef int8_t LHS;
    typedef int8_t RHS;

    static const int8_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_SUM_INT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class SumReduction<int16_t> {
  public:
    typedef int16_t LHS;
    typedef int16_t RHS;

    static const int16_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_SUM_INT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class SumReduction<int32_t> {
  public:
    typedef int32_t LHS;
    typedef int32_t RHS;

    static const int32_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_SUM_INT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class SumReduction<int64_t> {
  public:
    typedef int64_t LHS;
    typedef int64_t RHS;

    static const int64_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_SUM_INT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class SumReduction<uint8_t> {
  public:
    typedef uint8_t LHS;
    typedef uint8_t RHS;

    static const uint8_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_SUM_UINT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class SumReduction<uint16_t> {
  public:
    typedef uint16_t LHS;
    typedef uint16_t RHS;

    static const uint16_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_SUM_UINT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class SumReduction<uint32_t> {
  public:
    typedef uint32_t LHS;
    typedef uint32_t RHS;

    static const uint32_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_SUM_UINT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class SumReduction<uint64_t> {
  public:
    typedef uint64_t LHS;
    typedef uint64_t RHS;

    static const uint64_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_SUM_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_HALF
  template<>
  class SumReduction<__half> {
  public:
    typedef __half LHS;
    typedef __half RHS;

    static const __half identity;
    static const int REDOP_ID = LEGION_REDOP_SUM_FLOAT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif

  template<>
  class SumReduction<float> {
  public:
    typedef float LHS;
    typedef float RHS;

    static const float identity;
    static const int REDOP_ID = LEGION_REDOP_SUM_FLOAT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class SumReduction<double> {
  public:
    typedef double LHS;
    typedef double RHS;

    static const double identity;
    static const int REDOP_ID = LEGION_REDOP_SUM_FLOAT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<>
  class SumReduction<complex<__half> > {
  public:
    typedef complex<__half> LHS;
    typedef complex<__half> RHS;

    static const complex<__half> identity;
    static const int REDOP_ID = LEGION_REDOP_SUM_COMPLEX32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_HALF
   
  template<>
  class SumReduction<complex<float> > {
  public:
    typedef complex<float> LHS;
    typedef complex<float> RHS;

    static const complex<float> identity;
    static const int REDOP_ID = LEGION_REDOP_SUM_COMPLEX64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  // WARNING: This operator performs element-wise reductions on real and
  //          imaginary components, and thus has non-linearizable semantics.
  //          Users should be aware of this non-linearizability, which can
  //          lead to inconsistent results.
  template<>
  class SumReduction<complex<double> > {
  public:
    typedef complex<double> LHS;
    typedef complex<double> RHS;

    static const complex<double> identity;
    static const int REDOP_ID = LEGION_REDOP_SUM_COMPLEX128;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_COMPLEX

  template<typename T>
  class DiffReduction {
    // Empty definition
    // Specializations provided for each type
  };

  template<>
  class DiffReduction<int8_t> {
  public:
    typedef int8_t LHS;
    typedef int8_t RHS;

    static const int8_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_DIFF_INT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DiffReduction<int16_t> {
  public:
    typedef int16_t LHS;
    typedef int16_t RHS;

    static const int16_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_DIFF_INT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DiffReduction<int32_t> {
  public:
    typedef int32_t LHS;
    typedef int32_t RHS;

    static const int32_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_DIFF_INT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DiffReduction<int64_t> {
  public:
    typedef int64_t LHS;
    typedef int64_t RHS;

    static const int64_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_DIFF_INT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DiffReduction<uint8_t> {
  public:
    typedef uint8_t LHS;
    typedef uint8_t RHS;

    static const uint8_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_DIFF_UINT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DiffReduction<uint16_t> {
  public:
    typedef uint16_t LHS;
    typedef uint16_t RHS;

    static const uint16_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_DIFF_UINT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DiffReduction<uint32_t> {
  public:
    typedef uint32_t LHS;
    typedef uint32_t RHS;

    static const uint32_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_DIFF_UINT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DiffReduction<uint64_t> {
  public:
    typedef uint64_t LHS;
    typedef uint64_t RHS;

    static const uint64_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_DIFF_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_HALF
  template<>
  class DiffReduction<__half> {
  public:
    typedef __half LHS;
    typedef __half RHS;

    static const __half identity;
    static const int REDOP_ID = LEGION_REDOP_DIFF_FLOAT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif

  template<>
  class DiffReduction<float> {
  public:
    typedef float LHS;
    typedef float RHS;

    static const float identity;
    static const int REDOP_ID = LEGION_REDOP_DIFF_FLOAT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DiffReduction<double> {
  public:
    typedef double LHS;
    typedef double RHS;

    static const double identity;
    static const int REDOP_ID = LEGION_REDOP_DIFF_FLOAT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<>
  class DiffReduction<complex<__half> > {
  public:
    typedef complex<__half> LHS;
    typedef complex<__half> RHS;

    static const complex<__half> identity;
    static const int REDOP_ID = LEGION_REDOP_DIFF_COMPLEX32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_HALF
   
  template<>
  class DiffReduction<complex<float> > {
  public:
    typedef complex<float> LHS;
    typedef complex<float> RHS;

    static const complex<float> identity;
    static const int REDOP_ID = LEGION_REDOP_DIFF_COMPLEX64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_COMPLEX

  template<typename T>
  class ProdReduction {
    // Empty definition
    // Specializations provided for each type
  };

  template<>
  class ProdReduction<bool> {
  public:
    typedef bool LHS;
    typedef bool RHS;

    static const bool identity = true;
    static const int REDOP_ID = LEGION_REDOP_AND_BOOL;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class ProdReduction<int8_t> {
  public:
    typedef int8_t LHS;
    typedef int8_t RHS;

    static const int8_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_INT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class ProdReduction<int16_t> {
  public:
    typedef int16_t LHS;
    typedef int16_t RHS;

    static const int16_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_INT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class ProdReduction<int32_t> {
  public:
    typedef int32_t LHS;
    typedef int32_t RHS;

    static const int32_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_INT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class ProdReduction<int64_t> {
  public:
    typedef int64_t LHS;
    typedef int64_t RHS;

    static const int64_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_INT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class ProdReduction<uint8_t> {
  public:
    typedef uint8_t LHS;
    typedef uint8_t RHS;

    static const uint8_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_UINT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class ProdReduction<uint16_t> {
  public:
    typedef uint16_t LHS;
    typedef uint16_t RHS;

    static const uint16_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_UINT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class ProdReduction<uint32_t> {
  public:
    typedef uint32_t LHS;
    typedef uint32_t RHS;

    static const uint32_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_UINT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class ProdReduction<uint64_t> {
  public:
    typedef uint64_t LHS;
    typedef uint64_t RHS;

    static const uint64_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_HALF
  template<>
  class ProdReduction<__half> {
  public:
    typedef __half LHS;
    typedef __half RHS;

    static const __half identity;
    static const int REDOP_ID = LEGION_REDOP_PROD_FLOAT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif

  template<>
  class ProdReduction<float> {
  public:
    typedef float LHS;
    typedef float RHS;

    static const float identity;
    static const int REDOP_ID = LEGION_REDOP_PROD_FLOAT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class ProdReduction<double> {
  public:
    typedef double LHS;
    typedef double RHS;

    static const double identity;
    static const int REDOP_ID = LEGION_REDOP_PROD_FLOAT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_COMPLEX  
#ifdef LEGION_REDOP_HALF
  template<>
  class ProdReduction<complex<__half> > {
  public:
    typedef complex<__half> LHS;
    typedef complex<__half> RHS;

    static const complex<__half> identity;
    static const int REDOP_ID = LEGION_REDOP_DIFF_COMPLEX32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_HALF
   
  template<>
  class ProdReduction<complex<float> > {
  public:
    typedef complex<float> LHS;
    typedef complex<float> RHS;

    static const complex<float> identity;
    static const int REDOP_ID = LEGION_REDOP_DIFF_COMPLEX64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_COMPLEX

  template<typename T>
  class DivReduction {
    // Empty definition
    // Specializations provided for each type
  };

  template<>
  class DivReduction<int8_t> {
  public:
    typedef int8_t LHS;
    typedef int8_t RHS;

    static const int8_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_INT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DivReduction<int16_t> {
  public:
    typedef int16_t LHS;
    typedef int16_t RHS;

    static const int16_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_INT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DivReduction<int32_t> {
  public:
    typedef int32_t LHS;
    typedef int32_t RHS;

    static const int32_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_INT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DivReduction<int64_t> {
  public:
    typedef int64_t LHS;
    typedef int64_t RHS;

    static const int64_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_INT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DivReduction<uint8_t> {
  public:
    typedef uint8_t LHS;
    typedef uint8_t RHS;

    static const uint8_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_UINT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DivReduction<uint16_t> {
  public:
    typedef uint16_t LHS;
    typedef uint16_t RHS;

    static const uint16_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_UINT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DivReduction<uint32_t> {
  public:
    typedef uint32_t LHS;
    typedef uint32_t RHS;

    static const uint32_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_UINT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DivReduction<uint64_t> {
  public:
    typedef uint64_t LHS;
    typedef uint64_t RHS;

    static const uint64_t identity = 1;
    static const int REDOP_ID = LEGION_REDOP_PROD_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_HALF
  template<>
  class DivReduction<__half> {
  public:
    typedef __half LHS;
    typedef __half RHS;

    static const __half identity;
    static const int REDOP_ID = LEGION_REDOP_PROD_FLOAT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif

  template<>
  class DivReduction<float> {
  public:
    typedef float LHS;
    typedef float RHS;

    static const float identity;
    static const int REDOP_ID = LEGION_REDOP_PROD_FLOAT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class DivReduction<double> {
  public:
    typedef double LHS;
    typedef double RHS;

    static const double identity;
    static const int REDOP_ID = LEGION_REDOP_PROD_FLOAT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_COMPLEX  
#ifdef LEGION_REDOP_HALF
  template<>
  class DivReduction<complex<__half> > {
  public:
    typedef complex<__half> LHS;
    typedef complex<__half> RHS;

    static const complex<__half> identity;
    static const int REDOP_ID = LEGION_REDOP_DIFF_COMPLEX32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_HALF
   
  template<>
  class DivReduction<complex<float> > {
  public:
    typedef complex<float> LHS;
    typedef complex<float> RHS;

    static const complex<float> identity;
    static const int REDOP_ID = LEGION_REDOP_DIFF_COMPLEX64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_COMPLEX

  template<typename T>
  class MaxReduction {
    // Empty definition
    // Specializations provided for each type
  };

  template<>
  class MaxReduction<bool> {
  public:
    typedef bool LHS;
    typedef bool RHS;

    static const bool identity = false;
    static const int REDOP_ID = LEGION_REDOP_MAX_BOOL;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MaxReduction<int8_t> {
  public:
    typedef int8_t LHS;
    typedef int8_t RHS;

    static const int8_t identity = SCHAR_MIN;
    static const int REDOP_ID = LEGION_REDOP_MAX_INT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MaxReduction<int16_t> {
  public:
    typedef int16_t LHS;
    typedef int16_t RHS;

    static const int16_t identity = SHRT_MIN;
    static const int REDOP_ID = LEGION_REDOP_MAX_INT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MaxReduction<int32_t> {
  public:
    typedef int32_t LHS;
    typedef int32_t RHS;

    static const int32_t identity = INT_MIN;
    static const int REDOP_ID = LEGION_REDOP_MAX_INT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MaxReduction<int64_t> {
  public:
    typedef int64_t LHS;
    typedef int64_t RHS;

    static const int64_t identity = LLONG_MIN;
    static const int REDOP_ID = LEGION_REDOP_MAX_INT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MaxReduction<uint8_t> {
  public:
    typedef uint8_t LHS;
    typedef uint8_t RHS;

    static const uint8_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_MAX_UINT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MaxReduction<uint16_t> {
  public:
    typedef uint16_t LHS;
    typedef uint16_t RHS;

    static const uint16_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_MAX_UINT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MaxReduction<uint32_t> {
  public:
    typedef uint32_t LHS;
    typedef uint32_t RHS;

    static const uint32_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_MAX_UINT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MaxReduction<uint64_t> {
  public:
    typedef uint64_t LHS;
    typedef uint64_t RHS;

    static const uint64_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_MAX_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_HALF
  template<>
  class MaxReduction<__half> {
  public:
    typedef __half LHS;
    typedef __half RHS;

    static const __half identity;
    static const int REDOP_ID = LEGION_REDOP_MAX_FLOAT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif

  template<>
  class MaxReduction<float> {
  public:
    typedef float LHS;
    typedef float RHS;

    static const float identity;
    static const int REDOP_ID = LEGION_REDOP_MAX_FLOAT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MaxReduction<double> {
  public:
    typedef double LHS;
    typedef double RHS;

    static const double identity;
    static const int REDOP_ID = LEGION_REDOP_MAX_FLOAT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<>
  class MaxReduction<complex<__half> > {
  public:
    typedef complex<__half> LHS;
    typedef complex<__half> RHS;

    static const complex<__half> identity;
    static const int REDOP_ID = LEGION_REDOP_MAX_COMPLEX32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_HALF
  template<>
  class MaxReduction<complex<float> > {
  public:
    typedef complex<float> LHS;
    typedef complex<float> RHS;

    static const complex<float> identity;
    static const int REDOP_ID = LEGION_REDOP_MAX_COMPLEX64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
  // TODO: LEGION_REDOP_MAX_COMPLEX128,
#endif // LEGION_REDOP_COMPLEX

  template<typename T>
  class MinReduction {
    // Empty definition
    // Specializations provided for each type
  };

  template<>
  class MinReduction<bool> {
  public:
    typedef bool LHS;
    typedef bool RHS;

    static const bool identity = true;
    static const int REDOP_ID = LEGION_REDOP_MIN_BOOL;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MinReduction<int8_t> {
  public:
    typedef int8_t LHS;
    typedef int8_t RHS;

    static const int8_t identity = SCHAR_MAX;
    static const int REDOP_ID = LEGION_REDOP_MIN_INT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MinReduction<int16_t> {
  public:
    typedef int16_t LHS;
    typedef int16_t RHS;

    static const int16_t identity = SHRT_MAX;
    static const int REDOP_ID = LEGION_REDOP_MIN_INT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MinReduction<int32_t> {
  public:
    typedef int32_t LHS;
    typedef int32_t RHS;

    static const int32_t identity = INT_MAX;
    static const int REDOP_ID = LEGION_REDOP_MIN_INT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MinReduction<int64_t> {
  public:
    typedef int64_t LHS;
    typedef int64_t RHS;

    static const int64_t identity = LLONG_MAX;
    static const int REDOP_ID = LEGION_REDOP_MIN_INT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MinReduction<uint8_t> {
  public:
    typedef uint8_t LHS;
    typedef uint8_t RHS;

    static const uint8_t identity = UCHAR_MAX;
    static const int REDOP_ID = LEGION_REDOP_MIN_UINT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MinReduction<uint16_t> {
  public:
    typedef uint16_t LHS;
    typedef uint16_t RHS;

    static const uint16_t identity = USHRT_MAX;
    static const int REDOP_ID = LEGION_REDOP_MIN_UINT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MinReduction<uint32_t> {
  public:
    typedef uint32_t LHS;
    typedef uint32_t RHS;

    static const uint32_t identity = UINT_MAX;
    static const int REDOP_ID = LEGION_REDOP_MIN_UINT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MinReduction<uint64_t> {
  public:
    typedef uint64_t LHS;
    typedef uint64_t RHS;

    static const uint64_t identity = ULLONG_MAX;
    static const int REDOP_ID = LEGION_REDOP_MIN_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_HALF
  template<>
  class MinReduction<__half> {
  public:
    typedef __half LHS;
    typedef __half RHS;

    static const __half identity;
    static const int REDOP_ID = LEGION_REDOP_MIN_FLOAT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif

  template<>
  class MinReduction<float> {
  public:
    typedef float LHS;
    typedef float RHS;

    static const float identity;
    static const int REDOP_ID = LEGION_REDOP_MIN_FLOAT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class MinReduction<double> {
  public:
    typedef double LHS;
    typedef double RHS;

    static const double identity;
    static const int REDOP_ID = LEGION_REDOP_MIN_FLOAT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
  template<>
  class MinReduction<complex<__half> > {
  public:
    typedef complex<__half> LHS;
    typedef complex<__half> RHS;

    static const complex<__half> identity;
    static const int REDOP_ID = LEGION_REDOP_MIN_COMPLEX32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
#endif // LEGION_REDOP_HALF
  template<>
  class MinReduction<complex<float> > {
  public:
    typedef complex<float> LHS;
    typedef complex<float> RHS;

    static const complex<float> identity;
    static const int REDOP_ID = LEGION_REDOP_MIN_COMPLEX64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };
  // TODO: LEGION_REDOP_MAX_COMPLEX128,
#endif // LEGION_REDOP_COMPLEX

  template<typename T>
  class OrReduction {
    // Empty definition
    // Specializations provided for each type
  };

  template<>
  class OrReduction<int8_t> {
  public:
    typedef int8_t LHS;
    typedef int8_t RHS;

    static const int8_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_INT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class OrReduction<int16_t> {
  public:
    typedef int16_t LHS;
    typedef int16_t RHS;

    static const int16_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_INT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class OrReduction<int32_t> {
  public:
    typedef int32_t LHS;
    typedef int32_t RHS;

    static const int32_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_INT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class OrReduction<int64_t> {
  public:
    typedef int64_t LHS;
    typedef int64_t RHS;

    static const int64_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_INT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class OrReduction<uint8_t> {
  public:
    typedef uint8_t LHS;
    typedef uint8_t RHS;

    static const uint8_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class OrReduction<uint16_t> {
  public:
    typedef uint16_t LHS;
    typedef uint16_t RHS;

    static const uint16_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class OrReduction<uint32_t> {
  public:
    typedef uint32_t LHS;
    typedef uint32_t RHS;

    static const uint32_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class OrReduction<uint64_t> {
  public:
    typedef uint64_t LHS;
    typedef uint64_t RHS;

    static const uint64_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<typename T>
  class AndReduction {
    // Empty definition
    // Specializations provided for each type
  };

  template<>
  class AndReduction<int8_t> {
  public:
    typedef int8_t LHS;
    typedef int8_t RHS;

    static const int8_t identity = (int8_t)0xFF;
    static const int REDOP_ID = LEGION_REDOP_OR_INT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class AndReduction<int16_t> {
  public:
    typedef int16_t LHS;
    typedef int16_t RHS;

    static const int16_t identity = (int16_t)0xFFFF;
    static const int REDOP_ID = LEGION_REDOP_OR_INT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class AndReduction<int32_t> {
  public:
    typedef int32_t LHS;
    typedef int32_t RHS;

    static const int32_t identity = 0xFFFF;
    static const int REDOP_ID = LEGION_REDOP_OR_INT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class AndReduction<int64_t> {
  public:
    typedef int64_t LHS;
    typedef int64_t RHS;

    static const int64_t identity = 0xFFFFFFFFL;
    static const int REDOP_ID = LEGION_REDOP_OR_INT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class AndReduction<uint8_t> {
  public:
    typedef uint8_t LHS;
    typedef uint8_t RHS;

    static const uint8_t identity = 0xFFU;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class AndReduction<uint16_t> {
  public:
    typedef uint16_t LHS;
    typedef uint16_t RHS;

    static const uint16_t identity = 0xFFFFU;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class AndReduction<uint32_t> {
  public:
    typedef uint32_t LHS;
    typedef uint32_t RHS;

    static const uint32_t identity = 0xFFFFFFFFU;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class AndReduction<uint64_t> {
  public:
    typedef uint64_t LHS;
    typedef uint64_t RHS;

    static const uint64_t identity = 0xFFFFFFFFUL;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<typename T>
  class XorReduction {
    // Empty definition
    // Specializations provided for each type
  };

  template<>
  class XorReduction<bool> {
  public:
    typedef bool LHS;
    typedef bool RHS;

    static const bool identity = false;
    static const int REDOP_ID = LEGION_REDOP_OR_BOOL;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class XorReduction<int8_t> {
  public:
    typedef int8_t LHS;
    typedef int8_t RHS;

    static const int8_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_INT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class XorReduction<int16_t> {
  public:
    typedef int16_t LHS;
    typedef int16_t RHS;

    static const int16_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_INT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class XorReduction<int32_t> {
  public:
    typedef int32_t LHS;
    typedef int32_t RHS;

    static const int32_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_INT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class XorReduction<int64_t> {
  public:
    typedef int64_t LHS;
    typedef int64_t RHS;

    static const int64_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_INT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class XorReduction<uint8_t> {
  public:
    typedef uint8_t LHS;
    typedef uint8_t RHS;

    static const uint8_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT8;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class XorReduction<uint16_t> {
  public:
    typedef uint16_t LHS;
    typedef uint16_t RHS;

    static const uint16_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT16;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class XorReduction<uint32_t> {
  public:
    typedef uint32_t LHS;
    typedef uint32_t RHS;

    static const uint32_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT32;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  template<>
  class XorReduction<uint64_t> {
  public:
    typedef uint64_t LHS;
    typedef uint64_t RHS;

    static const uint64_t identity = 0;
    static const int REDOP_ID = LEGION_REDOP_OR_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

}; // namespace Legion

#include "legion_redop.inl"

#endif // __LEGION_REDOP_H__

