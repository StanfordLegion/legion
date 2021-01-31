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
class ArrayReductionOp : public Realm::ReductionOpUntyped {
public:
  ArrayReductionOp(unsigned n)
    : Realm::ReductionOpUntyped(sizeof(typename ELEM_REDOP::LHS) * n,
                                sizeof(typename ELEM_REDOP::RHS) * n,
// TODO: This will break if we change how a reduction list entry is laid out
#ifdef NEED_TO_FIX_REDUCTION_LISTS_FOR_DEPPART
                                sizeof(ptr_t) +
                                sizeof(typename ELEM_REDOP::RHS) * n,
#else
                                0,
#endif
                                true, true),
    N(n) {}

  virtual Realm::ReductionOpUntyped *clone(void) const
  {
    return new ArrayReductionOp<ELEM_REDOP>(N);
  }

  virtual void apply(void *lhs_ptr, const void *rhs_ptr, size_t count,
                     bool exclusive = false) const
  {
    typename ELEM_REDOP::LHS *lhs =
      static_cast<typename ELEM_REDOP::LHS *>(lhs_ptr);
    const typename ELEM_REDOP::RHS *rhs =
      static_cast<const typename ELEM_REDOP::RHS *>(rhs_ptr);
    size_t total_count = count * N;
    if (exclusive)
      for (size_t i = 0; i < total_count; i++)
        ELEM_REDOP::template apply<true>(lhs[i], rhs[i]);
    else
      for (size_t i = 0; i < total_count; i++)
        ELEM_REDOP::template apply<false>(lhs[i], rhs[i]);
  }

  virtual void apply_strided(void *lhs_ptr, const void *rhs_ptr,
                             off_t lhs_stride, off_t rhs_stride, size_t count,
                             bool exclusive = false) const
  {
    if (exclusive) {
      for (size_t i = 0; i < count; i++) {
        typename ELEM_REDOP::LHS *lhs =
          static_cast<typename ELEM_REDOP::LHS *>(lhs_ptr);
        const typename ELEM_REDOP::RHS *rhs =
          static_cast<const typename ELEM_REDOP::RHS *>(rhs_ptr);
        for (unsigned n = 0; n < N; ++n)
          ELEM_REDOP::template apply<true>(lhs[n], rhs[n]);
        lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
        rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
      }
    } else {
      for (size_t i = 0; i < count; i++) {
        typename ELEM_REDOP::LHS *lhs =
          static_cast<typename ELEM_REDOP::LHS *>(lhs_ptr);
        const typename ELEM_REDOP::RHS *rhs =
          static_cast<const typename ELEM_REDOP::RHS *>(rhs_ptr);
        for (unsigned n = 0; n < N; ++n)
          ELEM_REDOP::template apply<false>(lhs[n], rhs[n]);
        lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
        rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
      }
    }
  }

  virtual void fold(void *rhs1_ptr, const void *rhs2_ptr, size_t count,
                    bool exclusive = false) const
  {
    typename ELEM_REDOP::RHS *rhs1 =
      static_cast<typename ELEM_REDOP::RHS *>(rhs1_ptr);
    const typename ELEM_REDOP::RHS *rhs2 =
      static_cast<const typename ELEM_REDOP::RHS *>(rhs2_ptr);
    size_t total_count = count * N;
    if (exclusive)
      for (size_t i = 0; i < total_count; i++)
        ELEM_REDOP::template fold<true>(rhs1[i], rhs2[i]);
    else
      for (size_t i = 0; i < total_count; i++)
        ELEM_REDOP::template fold<false>(rhs1[i], rhs2[i]);
  }

  virtual void fold_strided(void *lhs_ptr, const void *rhs_ptr,
                            off_t lhs_stride, off_t rhs_stride, size_t count,
                            bool exclusive = false) const
  {
    if(exclusive) {
      for(size_t i = 0; i < count; i++) {
        typename ELEM_REDOP::LHS *lhs =
          static_cast<typename ELEM_REDOP::LHS *>(lhs_ptr);
        const typename ELEM_REDOP::RHS *rhs =
          static_cast<const typename ELEM_REDOP::RHS *>(rhs_ptr);
        for (unsigned n = 0; n < N; ++n)
          ELEM_REDOP::template fold<true>(lhs[n], rhs[n]);
        lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
        rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
      }
    } else {
      for(size_t i = 0; i < count; i++) {
        typename ELEM_REDOP::LHS *lhs =
          static_cast<typename ELEM_REDOP::LHS *>(lhs_ptr);
        const typename ELEM_REDOP::RHS *rhs =
          static_cast<const typename ELEM_REDOP::RHS *>(rhs_ptr);
        for (unsigned n = 0; n < N; ++n)
          ELEM_REDOP::template fold<false>(lhs[n], rhs[n]);
        lhs_ptr = static_cast<char *>(lhs_ptr) + lhs_stride;
        rhs_ptr = static_cast<const char *>(rhs_ptr) + rhs_stride;
      }
    }
  }

  virtual void init(void *ptr, size_t count) const
  {
    typename ELEM_REDOP::RHS *rhs_ptr =
      static_cast<typename ELEM_REDOP::RHS *>(ptr);
    size_t total_count = count * N;
    for (size_t i = 0; i < total_count; i++)
      *rhs_ptr++ = ELEM_REDOP::identity;
  }

#ifdef NEED_TO_FIX_REDUCTION_LISTS_FOR_DEPPART
  virtual void apply_list_entry(void *lhs_ptr, const void *entry_ptr, size_t count,
                                off_t ptr_offset, bool exclusive = false) const
  {
    // TODO: Implement this function
    assert(false);
  }

  virtual void fold_list_entry(void *rhs_ptr, const void *entry_ptr, size_t count,
                                off_t ptr_offset, bool exclusive = false) const
  {
    // TODO: Implement this function
    assert(false);
  }

  virtual void get_list_pointers(unsigned *ptrs, const void *entry_ptr, size_t count) const
  {
    // TODO: Implement this function
    assert(false);
  }
#endif

private:
  unsigned N;

public:
  static Realm::ReductionOpUntyped *create_array_reduction_op(unsigned array_size)
  {
    ArrayReductionOp<ELEM_REDOP> *redop =
      new ArrayReductionOp<ELEM_REDOP>(array_size);
    return redop;
  }
};

#define DECLARE_ARRAY_REDUCTION(REG, CLASS)                                  \
  extern "C"                                                                 \
  {                                                                          \
    void REG(legion_reduction_op_id_t redop_id, unsigned array_size,         \
             bool permit_duplicates)                                         \
    {                                                                        \
      ArrayReductionOp<CLASS> *op = new ArrayReductionOp<CLASS>(array_size); \
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
