/* Copyright 2019 Stanford University, NVIDIA Corporation
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

#ifdef LEGION_REDOP_HALF
#include "half.h"
#endif

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
#define COMPLEX_HALF 
#endif
#include "complex.h"
#endif

#include "legion/legion_config.h" // LEGION_MAX_APPLICATION_REDOP_ID

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

#define LEGION_REDOP_MAX_TYPES    16
#define LEGION_REDOP_VALUE(op_id, type_id) \
  LEGION_REDOP_BASE + op_id * LEGION_REDOP_MAX_TYPES + type_id

namespace Legion {

  enum LegionBuiltinRedops {
    LEGION_REDOP_BASE           = LEGION_MAX_APPLICATION_REDOP_ID,
    ////////////////////////////////////////
    // Sum reductions
    ////////////////////////////////////////
    LEGION_REDOP_OR_BOOL        = LEGION_REDOP_VALUE(0,0),
    LEGION_REDOP_SUM_INT16      = LEGION_REDOP_VALUE(0,1),
    LEGION_REDOP_SUM_INT32      = LEGION_REDOP_VALUE(0,2),
    LEGION_REDOP_SUM_INT64      = LEGION_REDOP_VALUE(0,3),
    LEGION_REDOP_SUM_UINT16     = LEGION_REDOP_VALUE(0,4),
    LEGION_REDOP_SUM_UINT32     = LEGION_REDOP_VALUE(0,5),
    LEGION_REDOP_SUM_UINT64     = LEGION_REDOP_VALUE(0,6),
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_SUM_FLOAT16    = LEGION_REDOP_VALUE(0,7),
#endif
    LEGION_REDOP_SUM_FLOAT32    = LEGION_REDOP_VALUE(0,8),
    LEGION_REDOP_SUM_FLOAT64    = LEGION_REDOP_VALUE(0,9),
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_SUM_COMPLEX32  = LEGION_REDOP_VALUE(0,10),
#endif
    LEGION_REDOP_SUM_COMPLEX64  = LEGION_REDOP_VALUE(0,11),
    // TODO: LEGION_REDOP_SUM_COMPLEX128,
#endif
    ////////////////////////////////////////
    // Difference reductions
    ////////////////////////////////////////
    // No difference for bools
    LEGION_REDOP_DIFF_INT16     = LEGION_REDOP_VALUE(1,1),
    LEGION_REDOP_DIFF_INT32     = LEGION_REDOP_VALUE(1,2),
    LEGION_REDOP_DIFF_INT64     = LEGION_REDOP_VALUE(1,3),
    LEGION_REDOP_DIFF_UINT16    = LEGION_REDOP_VALUE(1,4),
    LEGION_REDOP_DIFF_UINT32    = LEGION_REDOP_VALUE(1,5),
    LEGION_REDOP_DIFF_UINT64    = LEGION_REDOP_VALUE(1,6),
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_DIFF_FLOAT16   = LEGION_REDOP_VALUE(1,7),
#endif
    LEGION_REDOP_DIFF_FLOAT32   = LEGION_REDOP_VALUE(1,8),
    LEGION_REDOP_DIFF_FLOAT64   = LEGION_REDOP_VALUE(1,9),
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_DIFF_COMPLEX32 = LEGION_REDOP_VALUE(1,10),
#endif
    LEGION_REDOP_DIFF_COMPLEX64 = LEGION_REDOP_VALUE(1,11),
    // TODO: LEGION_REDOP_DIFF_COMPLEX128,
#endif
    ////////////////////////////////////////
    // Product reductions
    ////////////////////////////////////////
    LEGION_REDOP_AND_BOOL       = LEGION_REDOP_VALUE(2,0),
    LEGION_REDOP_PROD_INT16     = LEGION_REDOP_VALUE(2,1),
    LEGION_REDOP_PROD_INT32     = LEGION_REDOP_VALUE(2,2),
    LEGION_REDOP_PROD_INT64     = LEGION_REDOP_VALUE(2,3),
    LEGION_REDOP_PROD_UINT16    = LEGION_REDOP_VALUE(2,4),
    LEGION_REDOP_PROD_UINT32    = LEGION_REDOP_VALUE(2,5),
    LEGION_REDOP_PROD_UINT64    = LEGION_REDOP_VALUE(2,6),
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_PROD_FLOAT16   = LEGION_REDOP_VALUE(2,7),
#endif
    LEGION_REDOP_PROD_FLOAT32   = LEGION_REDOP_VALUE(2,8),
    LEGION_REDOP_PROD_FLOAT64   = LEGION_REDOP_VALUE(2,9),
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_PROD_COMPLEX32 = LEGION_REDOP_VALUE(2,10),
#endif
    LEGION_REDOP_PROD_COMPLEX64 = LEGION_REDOP_VALUE(2,11),
    // TODO: LEGION_REDOP_PROD_COMPLEX128,
#endif
    ////////////////////////////////////////
    // Division reductions
    ////////////////////////////////////////
    // No division for bools
    LEGION_REDOP_DIV_INT16      = LEGION_REDOP_VALUE(3,1),
    LEGION_REDOP_DIV_INT32      = LEGION_REDOP_VALUE(3,2),
    LEGION_REDOP_DIV_INT64      = LEGION_REDOP_VALUE(3,3),
    LEGION_REDOP_DIV_UINT16     = LEGION_REDOP_VALUE(3,4),
    LEGION_REDOP_DIV_UINT32     = LEGION_REDOP_VALUE(3,5),
    LEGION_REDOP_DIV_UINT64     = LEGION_REDOP_VALUE(3,6),
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_DIV_FLOAT16    = LEGION_REDOP_VALUE(3,7),
#endif
    LEGION_REDOP_DIV_FLOAT32    = LEGION_REDOP_VALUE(3,8),
    LEGION_REDOP_DIV_FLOAT64    = LEGION_REDOP_VALUE(3,9),
#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_DIV_COMPLEX32  = LEGION_REDOP_VALUE(3,10),
#endif
    LEGION_REDOP_DIV_COMPLEX64  = LEGION_REDOP_VALUE(3,11),
    // TODO: LEGION_REDOP_DIV_COMPLEX128,
#endif
    ////////////////////////////////////////
    // Max reductions
    ////////////////////////////////////////
    LEGION_REDOP_MAX_BOOL       = LEGION_REDOP_VALUE(4,0),
    LEGION_REDOP_MAX_INT16      = LEGION_REDOP_VALUE(4,1),
    LEGION_REDOP_MAX_INT32      = LEGION_REDOP_VALUE(4,2),
    LEGION_REDOP_MAX_INT64      = LEGION_REDOP_VALUE(4,3),
    LEGION_REDOP_MAX_UINT16     = LEGION_REDOP_VALUE(4,4),
    LEGION_REDOP_MAX_UINT32     = LEGION_REDOP_VALUE(4,5),
    LEGION_REDOP_MAX_UINT64     = LEGION_REDOP_VALUE(4,6),
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_MAX_FLOAT16    = LEGION_REDOP_VALUE(4,7),
#endif
    LEGION_REDOP_MAX_FLOAT32    = LEGION_REDOP_VALUE(4,8),
    LEGION_REDOP_MAX_FLOAT64    = LEGION_REDOP_VALUE(4,9),
    // No definitions of max for complex types
    ////////////////////////////////////////
    // Min reductions
    ////////////////////////////////////////
    LEGION_REDOP_MIN_BOOL       = LEGION_REDOP_VALUE(5,0),
    LEGION_REDOP_MIN_INT16      = LEGION_REDOP_VALUE(5,1),
    LEGION_REDOP_MIN_INT32      = LEGION_REDOP_VALUE(5,2),
    LEGION_REDOP_MIN_INT64      = LEGION_REDOP_VALUE(5,3),
    LEGION_REDOP_MIN_UINT16     = LEGION_REDOP_VALUE(5,4),
    LEGION_REDOP_MIN_UINT32     = LEGION_REDOP_VALUE(5,5),
    LEGION_REDOP_MIN_UINT64     = LEGION_REDOP_VALUE(5,6),
#ifdef LEGION_REDOP_HALF
    LEGION_REDOP_MIN_FLOAT16    = LEGION_REDOP_VALUE(5,7),
#endif
    LEGION_REDOP_MIN_FLOAT32    = LEGION_REDOP_VALUE(5,8),
    LEGION_REDOP_MIN_FLOAT64    = LEGION_REDOP_VALUE(5,9),
    // No definitions of min for complex types
  };

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
#endif // LEGION_REDOP_COMPLEX

  template<typename T>
  class DiffReduction {
    // Empty definition
    // Specializations provided for each type
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

}; // namespace Legion

#include "legion_redop.inl"

#endif // __LEGION_REDOP_H__

