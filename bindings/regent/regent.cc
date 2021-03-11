/* Copyright 2021 Stanford University
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

#include <cfloat>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <limits>
#include <vector>

#define LEGION_ENABLE_C_BINDINGS
#include "legion.h"
#include "legion/legion_c_util.h"
#include "legion/legion_redop.h"
#include "realm/redop.h"

#include "regent.h"

using namespace Legion;

typedef Realm::Point<1,coord_t> Point1D;
typedef Realm::Point<2,coord_t> Point2D;
typedef Realm::Point<3,coord_t> Point3D;
typedef CObjectWrapper::ArrayAccessor1D ArrayAccessor1D;
typedef CObjectWrapper::ArrayAccessor2D ArrayAccessor2D;
typedef CObjectWrapper::ArrayAccessor3D ArrayAccessor3D;

template <class ELEM_REDOP>
struct ArrayReductionOp : public Realm::ReductionOpUntyped {
  unsigned N;
  typename ELEM_REDOP::RHS identity_val[1 /*really N*/];

protected:
  template <bool EXCL>
  static void cpu_apply_wrapper(void *lhs_ptr, size_t lhs_stride,
                                const void *rhs_ptr, size_t rhs_stride,
                                size_t count, const void *userdata)
  {
    unsigned N = *static_cast<const unsigned *>(userdata);
    for (size_t i = 0; i < count; i++) {
      typename ELEM_REDOP::LHS *lhs =
        static_cast<typename ELEM_REDOP::LHS *>(lhs_ptr);
      const typename ELEM_REDOP::RHS *rhs =
        static_cast<const typename ELEM_REDOP::RHS *>(rhs_ptr);
      for (unsigned n = 0; n < N; ++n)
        ELEM_REDOP::template apply<EXCL>(lhs[n], rhs[n]);
      lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
      rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
    }
  }

  template <bool EXCL>
  static void cpu_fold_wrapper(void *rhs1_ptr, size_t rhs1_stride,
                               const void *rhs2_ptr, size_t rhs2_stride,
                               size_t count, const void *userdata)
  {
    unsigned N = *static_cast<const unsigned *>(userdata);
    for(size_t i = 0; i < count; i++) {
      typename ELEM_REDOP::RHS *rhs1 =
        static_cast<typename ELEM_REDOP::RHS *>(rhs1_ptr);
      const typename ELEM_REDOP::RHS *rhs2 =
        static_cast<const typename ELEM_REDOP::RHS *>(rhs2_ptr);
      for (unsigned n = 0; n < N; ++n)
        ELEM_REDOP::template fold<EXCL>(rhs1[n], rhs2[n]);
      rhs1_ptr = static_cast<char *>(rhs1_ptr) + rhs1_stride;
      rhs2_ptr = static_cast<const char *>(rhs2_ptr) + rhs2_stride;
    }
  }

  ArrayReductionOp(unsigned n)
  {
    sizeof_this = (sizeof(ArrayReductionOp<ELEM_REDOP>) +
                   ((n - 1) * sizeof(typename ELEM_REDOP::RHS)));
    sizeof_lhs = sizeof(typename ELEM_REDOP::LHS) * n;
    sizeof_rhs = sizeof(typename ELEM_REDOP::RHS) * n;
    sizeof_userdata = sizeof(unsigned);
    identity = identity_val;
    userdata = &N;
    cpu_apply_excl_fn = &cpu_apply_wrapper<true>;
    cpu_apply_nonexcl_fn = &cpu_apply_wrapper<false>;
    cpu_fold_excl_fn = &cpu_fold_wrapper<true>;
    cpu_fold_nonexcl_fn = &cpu_fold_wrapper<false>;
    N = n;
    for(unsigned i = 0; i < n; i++)
      identity_val[i] = ELEM_REDOP::identity;
  }

public:
  static ArrayReductionOp<ELEM_REDOP> *create_array_reduction_op(unsigned array_size)
  {
    size_t bytes = (sizeof(ArrayReductionOp<ELEM_REDOP>) +
                    ((array_size - 1) * sizeof(typename ELEM_REDOP::RHS)));
    void *ptr = malloc(bytes);
    assert(ptr);
    return new(ptr) ArrayReductionOp<ELEM_REDOP>(array_size);
  }
};

#define DECLARE_ARRAY_REDUCTION(REG, CLASS)                                  \
  extern "C"                                                                 \
  {                                                                          \
    void REG(legion_reduction_op_id_t redop_id, unsigned array_size,         \
             bool permit_duplicates)                                         \
    {                                                                        \
      ArrayReductionOp<CLASS> *op = ArrayReductionOp<CLASS>::create_array_reduction_op(array_size); \
      Runtime::register_reduction_op(redop_id, op, NULL, NULL,               \
                                     permit_duplicates);                     \
    }                                                                        \
  }

DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_float  , SumReduction<float>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_double , SumReduction<double>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_int16  , SumReduction<int16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_int32  , SumReduction<int32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_int64  , SumReduction<int64_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_uint16 , SumReduction<uint16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_uint32 , SumReduction<uint32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_uint64 , SumReduction<uint64_t>)
#ifdef LEGION_REDOP_COMPLEX
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_complex64 , SumReduction<complex<float> >)
#endif

DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_float  , DiffReduction<float>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_double , DiffReduction<double>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_int16  , DiffReduction<int16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_int32  , DiffReduction<int32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_int64  , DiffReduction<int64_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_uint16 , DiffReduction<uint16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_uint32 , DiffReduction<uint32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_uint64 , DiffReduction<uint64_t>)
#ifdef LEGION_REDOP_COMPLEX
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_complex64 , DiffReduction<complex<float> >)
#endif

DECLARE_ARRAY_REDUCTION(register_array_reduction_times_float  , ProdReduction<float>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_double , ProdReduction<double>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_int16  , ProdReduction<int16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_int32  , ProdReduction<int32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_int64  , ProdReduction<int64_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_uint16 , ProdReduction<uint16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_uint32 , ProdReduction<uint32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_uint64 , ProdReduction<uint64_t>)
#ifdef LEGION_REDOP_COMPLEX
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_complex64 , ProdReduction<complex<float> >)
#endif

DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_float  , DivReduction<float>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_double , DivReduction<double>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_int16  , DivReduction<int16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_int32  , DivReduction<int32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_int64  , DivReduction<int64_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_uint16 , DivReduction<uint16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_uint32 , DivReduction<uint32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_uint64 , DivReduction<uint64_t>)
#ifdef LEGION_REDOP_COMPLEX
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_complex64 , DivReduction<complex<float> >)
#endif

DECLARE_ARRAY_REDUCTION(register_array_reduction_max_float  , MaxReduction<float>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_double , MaxReduction<double>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_int16  , MaxReduction<int16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_int32  , MaxReduction<int32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_int64  , MaxReduction<int64_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_uint16 , MaxReduction<uint16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_uint32 , MaxReduction<uint32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_uint64 , MaxReduction<uint64_t>)

DECLARE_ARRAY_REDUCTION(register_array_reduction_min_float  , MinReduction<float>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_double , MinReduction<double>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_int16  , MinReduction<int16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_int32  , MinReduction<int32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_int64  , MinReduction<int64_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_uint16 , MinReduction<uint16_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_uint32 , MinReduction<uint32_t>)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_uint64 , MinReduction<uint64_t>)

#undef DECLARE_ARRAY_REDUCTION
