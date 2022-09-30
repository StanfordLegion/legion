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
#if defined (__CUDACC__) || defined (__HIPCC__)
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

#include "legion/legion_config.h"

#include <cstdint>

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
    static constexpr int REDOP_ID = LEGION_REDOP_OR_BOOL;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_INT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_INT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_INT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_INT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_UINT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_UINT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_UINT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_UINT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_FLOAT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_FLOAT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_FLOAT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_COMPLEX32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_COMPLEX64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_SUM_COMPLEX128;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_INT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_INT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_INT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_INT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_UINT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_UINT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_UINT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_UINT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_FLOAT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_FLOAT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_FLOAT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_COMPLEX32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIFF_COMPLEX64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_AND_BOOL;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_INT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_INT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_INT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_INT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_UINT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_UINT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_UINT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_UINT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_FLOAT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_FLOAT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_FLOAT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_COMPLEX32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_PROD_COMPLEX64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_INT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_INT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_INT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_INT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_UINT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_UINT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_UINT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_UINT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_FLOAT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_FLOAT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_FLOAT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_COMPLEX32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_DIV_COMPLEX64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_BOOL;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_INT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_INT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_INT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_INT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_UINT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_UINT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_UINT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_UINT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_FLOAT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_FLOAT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MAX_FLOAT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_BOOL;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_INT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_INT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_INT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_INT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_UINT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_UINT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_UINT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_UINT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_FLOAT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_FLOAT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_MIN_FLOAT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

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
    static constexpr int REDOP_ID = LEGION_REDOP_OR_INT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_OR_INT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_OR_INT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_OR_INT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_OR_UINT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_OR_UINT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_OR_UINT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_OR_UINT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_AND_INT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_AND_INT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_AND_INT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_AND_INT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_AND_UINT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_AND_UINT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_AND_UINT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_AND_UINT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_XOR_BOOL;

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
    static constexpr int REDOP_ID = LEGION_REDOP_XOR_INT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_XOR_INT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_XOR_INT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_XOR_INT64;

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
    static constexpr int REDOP_ID = LEGION_REDOP_XOR_UINT8;

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
    static constexpr int REDOP_ID = LEGION_REDOP_XOR_UINT16;

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
    static constexpr int REDOP_ID = LEGION_REDOP_XOR_UINT32;

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
    static constexpr int REDOP_ID = LEGION_REDOP_XOR_UINT64;

    template<bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS &lhs, RHS rhs);
    template<bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS &rhs1, RHS rhs2);
  };

  // Depending on configuration, we may instantiate these reduction ops in a
  //  one or more different ways, so define a helper macro that lists all the
  //  ops to be instantiated
#define LEGION_REDOP_LIST_BASE(__op__) \
      /* Sum Reductions */ \
      __op__(LEGION_REDOP_SUM_INT8, SumReduction<int8_t>) \
      __op__(LEGION_REDOP_SUM_INT16, SumReduction<int16_t>) \
      __op__(LEGION_REDOP_SUM_INT32, SumReduction<int32_t>) \
      __op__(LEGION_REDOP_SUM_INT64, SumReduction<int64_t>) \
      __op__(LEGION_REDOP_SUM_UINT8, SumReduction<uint8_t>) \
      __op__(LEGION_REDOP_SUM_UINT16, SumReduction<uint16_t>) \
      __op__(LEGION_REDOP_SUM_UINT32, SumReduction<uint32_t>) \
      __op__(LEGION_REDOP_SUM_UINT64, SumReduction<uint64_t>) \
      __op__(LEGION_REDOP_SUM_FLOAT32, SumReduction<float>) \
      __op__(LEGION_REDOP_SUM_FLOAT64, SumReduction<double>) \
      /* Difference Reductions */ \
      __op__(LEGION_REDOP_DIFF_INT8, DiffReduction<int8_t>) \
      __op__(LEGION_REDOP_DIFF_INT16, DiffReduction<int16_t>) \
      __op__(LEGION_REDOP_DIFF_INT32, DiffReduction<int32_t>) \
      __op__(LEGION_REDOP_DIFF_INT64, DiffReduction<int64_t>) \
      __op__(LEGION_REDOP_DIFF_UINT8, DiffReduction<uint8_t>) \
      __op__(LEGION_REDOP_DIFF_UINT16, DiffReduction<uint16_t>) \
      __op__(LEGION_REDOP_DIFF_UINT32, DiffReduction<uint32_t>) \
      __op__(LEGION_REDOP_DIFF_UINT64, DiffReduction<uint64_t>) \
      __op__(LEGION_REDOP_DIFF_FLOAT32, DiffReduction<float>) \
      __op__(LEGION_REDOP_DIFF_FLOAT64, DiffReduction<double>) \
      /* Product Reductions */ \
      __op__(LEGION_REDOP_PROD_INT8, ProdReduction<int8_t>) \
      __op__(LEGION_REDOP_PROD_INT16, ProdReduction<int16_t>) \
      __op__(LEGION_REDOP_PROD_INT32, ProdReduction<int32_t>) \
      __op__(LEGION_REDOP_PROD_INT64, ProdReduction<int64_t>) \
      __op__(LEGION_REDOP_PROD_UINT8, ProdReduction<uint8_t>) \
      __op__(LEGION_REDOP_PROD_UINT16, ProdReduction<uint16_t>) \
      __op__(LEGION_REDOP_PROD_UINT32, ProdReduction<uint32_t>) \
      __op__(LEGION_REDOP_PROD_UINT64, ProdReduction<uint64_t>) \
      __op__(LEGION_REDOP_PROD_FLOAT32, ProdReduction<float>) \
      __op__(LEGION_REDOP_PROD_FLOAT64, ProdReduction<double>) \
      /* Divide Reductions */ \
      __op__(LEGION_REDOP_DIV_INT8, DivReduction<int8_t>) \
      __op__(LEGION_REDOP_DIV_INT16, DivReduction<int16_t>) \
      __op__(LEGION_REDOP_DIV_INT32, DivReduction<int32_t>) \
      __op__(LEGION_REDOP_DIV_INT64, DivReduction<int64_t>) \
      __op__(LEGION_REDOP_DIV_UINT8, DivReduction<uint8_t>) \
      __op__(LEGION_REDOP_DIV_UINT16, DivReduction<uint16_t>) \
      __op__(LEGION_REDOP_DIV_UINT32, DivReduction<uint32_t>) \
      __op__(LEGION_REDOP_DIV_UINT64, DivReduction<uint64_t>) \
      __op__(LEGION_REDOP_DIV_FLOAT32, DivReduction<float>) \
      __op__(LEGION_REDOP_DIV_FLOAT64, DivReduction<double>) \
      /* Max Reductions */ \
      __op__(LEGION_REDOP_MAX_BOOL, MaxReduction<bool>) \
      __op__(LEGION_REDOP_MAX_INT8, MaxReduction<int8_t>) \
      __op__(LEGION_REDOP_MAX_INT16, MaxReduction<int16_t>) \
      __op__(LEGION_REDOP_MAX_INT32, MaxReduction<int32_t>) \
      __op__(LEGION_REDOP_MAX_INT64, MaxReduction<int64_t>) \
      __op__(LEGION_REDOP_MAX_UINT8, MaxReduction<uint8_t>) \
      __op__(LEGION_REDOP_MAX_UINT16, MaxReduction<uint16_t>) \
      __op__(LEGION_REDOP_MAX_UINT32, MaxReduction<uint32_t>) \
      __op__(LEGION_REDOP_MAX_UINT64, MaxReduction<uint64_t>) \
      __op__(LEGION_REDOP_MAX_FLOAT32, MaxReduction<float>) \
      __op__(LEGION_REDOP_MAX_FLOAT64, MaxReduction<double>) \
      /* Min Reductions */ \
      __op__(LEGION_REDOP_MIN_BOOL, MinReduction<bool>) \
      __op__(LEGION_REDOP_MIN_INT8, MinReduction<int8_t>) \
      __op__(LEGION_REDOP_MIN_INT16, MinReduction<int16_t>) \
      __op__(LEGION_REDOP_MIN_INT32, MinReduction<int32_t>) \
      __op__(LEGION_REDOP_MIN_INT64, MinReduction<int64_t>) \
      __op__(LEGION_REDOP_MIN_UINT8, MinReduction<uint8_t>) \
      __op__(LEGION_REDOP_MIN_UINT16, MinReduction<uint16_t>) \
      __op__(LEGION_REDOP_MIN_UINT32, MinReduction<uint32_t>) \
      __op__(LEGION_REDOP_MIN_UINT64, MinReduction<uint64_t>) \
      __op__(LEGION_REDOP_MIN_FLOAT32, MinReduction<float>) \
      __op__(LEGION_REDOP_MIN_FLOAT64, MinReduction<double>) \
      /* Bitwise-OR Reductions */ \
      __op__(LEGION_REDOP_OR_BOOL, SumReduction<bool>) \
      __op__(LEGION_REDOP_OR_INT8, OrReduction<int8_t>) \
      __op__(LEGION_REDOP_OR_INT16, OrReduction<int16_t>) \
      __op__(LEGION_REDOP_OR_INT32, OrReduction<int32_t>) \
      __op__(LEGION_REDOP_OR_INT64, OrReduction<int64_t>) \
      __op__(LEGION_REDOP_OR_UINT8, OrReduction<uint8_t>) \
      __op__(LEGION_REDOP_OR_UINT16, OrReduction<uint16_t>) \
      __op__(LEGION_REDOP_OR_UINT32, OrReduction<uint32_t>) \
      __op__(LEGION_REDOP_OR_UINT64, OrReduction<uint64_t>) \
      /* Bitwise-AND Reductions */ \
      __op__(LEGION_REDOP_AND_BOOL, ProdReduction<bool>) \
      __op__(LEGION_REDOP_AND_INT8, AndReduction<int8_t>) \
      __op__(LEGION_REDOP_AND_INT16, AndReduction<int16_t>) \
      __op__(LEGION_REDOP_AND_INT32, AndReduction<int32_t>) \
      __op__(LEGION_REDOP_AND_INT64, AndReduction<int64_t>) \
      __op__(LEGION_REDOP_AND_UINT8, AndReduction<uint8_t>) \
      __op__(LEGION_REDOP_AND_UINT16, AndReduction<uint16_t>) \
      __op__(LEGION_REDOP_AND_UINT32, AndReduction<uint32_t>) \
      __op__(LEGION_REDOP_AND_UINT64, AndReduction<uint64_t>) \
      /* Bitwise-XOR Reductions */ \
      __op__(LEGION_REDOP_XOR_BOOL, XorReduction<bool>) \
      __op__(LEGION_REDOP_XOR_INT8, XorReduction<int8_t>) \
      __op__(LEGION_REDOP_XOR_INT16, XorReduction<int16_t>) \
      __op__(LEGION_REDOP_XOR_INT32, XorReduction<int32_t>) \
      __op__(LEGION_REDOP_XOR_INT64, XorReduction<int64_t>) \
      __op__(LEGION_REDOP_XOR_UINT8, XorReduction<uint8_t>) \
      __op__(LEGION_REDOP_XOR_UINT16, XorReduction<uint16_t>) \
      __op__(LEGION_REDOP_XOR_UINT32, XorReduction<uint32_t>) \
      __op__(LEGION_REDOP_XOR_UINT64, XorReduction<uint64_t>)

#ifdef LEGION_REDOP_HALF
  #define LEGION_REDOP_LIST_HALF(__op__)    \
      __op__(LEGION_REDOP_SUM_FLOAT16, SumReduction<__half>) \
      __op__(LEGION_REDOP_DIFF_FLOAT16, DiffReduction<__half>) \
      __op__(LEGION_REDOP_PROD_FLOAT16, ProdReduction<__half>) \
      __op__(LEGION_REDOP_DIV_FLOAT16, DivReduction<__half>) \
      __op__(LEGION_REDOP_MAX_FLOAT16, MaxReduction<__half>) \
      __op__(LEGION_REDOP_MIN_FLOAT16, MinReduction<__half>)
#else
  #define LEGION_REDOP_LIST_HALF(__op__)
#endif
#ifdef LEGION_REDOP_COMPLEX
  #define LEGION_REDOP_LIST_COMPLEX(__op__) \
      __op__(LEGION_REDOP_SUM_COMPLEX64, SumReduction<complex<float> >) \
      __op__(LEGION_REDOP_DIFF_COMPLEX64, DiffReduction<complex<float> >) \
      __op__(LEGION_REDOP_PROD_COMPLEX64, ProdReduction<complex<float> >) \
      __op__(LEGION_REDOP_DIV_COMPLEX64, DivReduction<complex<float> >) \
      __op__(LEGION_REDOP_SUM_COMPLEX128, SumReduction<complex<double> >)

  #ifdef LEGION_REDOP_HALF
    #define LEGION_REDOP_LIST_HALF_COMPLEX(__op__)    \
      __op__(LEGION_REDOP_SUM_COMPLEX32, SumReduction<complex<__half> >) \
      __op__(LEGION_REDOP_DIFF_COMPLEX32, DiffReduction<complex<__half> >) \
      __op__(LEGION_REDOP_PROD_COMPLEX32, ProdReduction<complex<__half> >) \
      __op__(LEGION_REDOP_DIV_COMPLEX32, DivReduction<complex<__half> >)
  #else
    #define LEGION_REDOP_LIST_HALF_COMPLEX(__op__)
  #endif
#else
  #define LEGION_REDOP_LIST_COMPLEX(__op__)
  #define LEGION_REDOP_LIST_HALF_COMPLEX(__op__)
#endif

#define LEGION_REDOP_LIST(__op__) \
  LEGION_REDOP_LIST_BASE(__op__) \
  LEGION_REDOP_LIST_HALF(__op__) \
  LEGION_REDOP_LIST_COMPLEX(__op__) \
  LEGION_REDOP_LIST_HALF_COMPLEX(__op__)

}; // namespace Legion

#include "legion_redop.inl"

#endif // __LEGION_REDOP_H__

