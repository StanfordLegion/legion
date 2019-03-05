/* Copyright 2019 Stanford University
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
#include "realm/redop.h"

#include "regent.h"

using namespace Legion;

typedef Realm::Point<1,coord_t> Point1D;
typedef Realm::Point<2,coord_t> Point2D;
typedef Realm::Point<3,coord_t> Point3D;
typedef CObjectWrapper::ArrayAccessor1D ArrayAccessor1D;
typedef CObjectWrapper::ArrayAccessor2D ArrayAccessor2D;
typedef CObjectWrapper::ArrayAccessor3D ArrayAccessor3D;

#define ADD(x, y) ((x) + (y))
#define SUB(x, y) ((x) - (y))
#define MUL(x, y) ((x) * (y))
#define DIV(x, y) ((x) / (y))

// Pre-defined reduction operators
#define DECLARE_REDUCTION(REG, SRED, SRED_DP, RED, RED_DP, CLASS, T, U, APPLY_OP, FOLD_OP, ID) \
  class CLASS {                                                         \
  public:                                                               \
  typedef T LHS, RHS;                                                   \
  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);       \
  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);      \
  static const T identity;                                              \
  };                                                                    \
                                                                        \
  const T CLASS::identity = ID;                                         \
                                                                        \
  template <>                                                           \
  void CLASS::apply<true>(LHS &lhs, RHS rhs)                            \
  {                                                                     \
    lhs = APPLY_OP(lhs, rhs);                                           \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::apply<false>(LHS &lhs, RHS rhs)                           \
  {                                                                     \
    volatile U *target = (U *)&(lhs);                                   \
    union { U as_U; T as_T; } oldval, newval;                           \
    do {                                                                \
      oldval.as_U = *target;                                            \
      newval.as_T = APPLY_OP(oldval.as_T, rhs);                         \
    } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<true>(RHS &rhs1, RHS rhs2)                           \
  {                                                                     \
    rhs1 = FOLD_OP(rhs1, rhs2);                                         \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<false>(RHS &rhs1, RHS rhs2)                          \
  {                                                                     \
    volatile U *target = (U *)&rhs1;                                    \
    union { U as_U; T as_T; } oldval, newval;                           \
    do {                                                                \
      oldval.as_U = *target;                                            \
      newval.as_T = FOLD_OP(oldval.as_T, rhs2);                         \
    } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
  }                                                                     \
                                                                        \
  extern "C"                                                            \
  {                                                                     \
  void REG(legion_reduction_op_id_t redop_id)                           \
  {                                                                     \
    Runtime::register_reduction_op<CLASS>(redop_id);                    \
  }                                                                     \
  void SRED(legion_accessor_array_1d_t accessor_,                       \
           legion_ptr_t ptr_, T value)                                  \
  {                                                                     \
    ArrayAccessor1D* accessor = CObjectWrapper::unwrap(accessor_);      \
    ptr_t ptr = CObjectWrapper::unwrap(ptr_);                           \
    CLASS::fold<false>(*(T*)(accessor->ptr(ptr.value)), value);         \
  }                                                                     \
  void SRED_DP##_1d(legion_accessor_array_1d_t accessor_,               \
                  legion_point_1d_t p_, T value)                        \
  {                                                                     \
    ArrayAccessor1D* accessor = CObjectWrapper::unwrap(accessor_);      \
    Point1D p = CObjectWrapper::unwrap(p_);                             \
    CLASS::fold<false>(*(T*)(accessor->ptr(p)), value);                 \
  }                                                                     \
  void SRED_DP##_2d(legion_accessor_array_2d_t accessor_,               \
                  legion_point_2d_t p_, T value)                        \
  {                                                                     \
    ArrayAccessor2D* accessor = CObjectWrapper::unwrap(accessor_);      \
    Point2D p = CObjectWrapper::unwrap(p_);                             \
    CLASS::fold<false>(*(T*)(accessor->ptr(p)), value);                 \
  }                                                                     \
  void SRED_DP##_3d(legion_accessor_array_3d_t accessor_,               \
                  legion_point_3d_t p_, T value)                        \
  {                                                                     \
    ArrayAccessor3D* accessor = CObjectWrapper::unwrap(accessor_);      \
    Point3D p = CObjectWrapper::unwrap(p_);                             \
    CLASS::fold<false>(*(T*)(accessor->ptr(p)), value);                 \
  }                                                                     \
  void RED(legion_accessor_array_1d_t accessor_,                        \
           legion_ptr_t ptr_, T value)                                  \
  {                                                                     \
    ArrayAccessor1D* accessor = CObjectWrapper::unwrap(accessor_);      \
    ptr_t ptr = CObjectWrapper::unwrap(ptr_);                           \
    CLASS::fold<true>(*(T*)(accessor->ptr(ptr.value)), value);          \
  }                                                                     \
  void RED_DP##_1d(legion_accessor_array_1d_t accessor_,                \
                 legion_point_1d_t p_, T value)                         \
  {                                                                     \
    ArrayAccessor1D* accessor = CObjectWrapper::unwrap(accessor_);      \
    Point1D p = CObjectWrapper::unwrap(p_);                             \
    CLASS::fold<true>(*(T*)(accessor->ptr(p)), value);                  \
  }                                                                     \
  void RED_DP##_2d(legion_accessor_array_2d_t accessor_,                \
                 legion_point_2d_t p_, T value)                         \
  {                                                                     \
    ArrayAccessor2D* accessor = CObjectWrapper::unwrap(accessor_);      \
    Point2D p = CObjectWrapper::unwrap(p_);                             \
    CLASS::fold<true>(*(T*)(accessor->ptr(p)), value);                  \
  }                                                                     \
  void RED_DP##_3d(legion_accessor_array_3d_t accessor_,                \
                 legion_point_3d_t p_, T value)                         \
  {                                                                     \
    ArrayAccessor3D* accessor = CObjectWrapper::unwrap(accessor_);      \
    Point3D p = CObjectWrapper::unwrap(p_);                             \
    CLASS::fold<true>(*(T*)(accessor->ptr(p)), value);                  \
  }                                                                     \
  }                                                                     \

DECLARE_REDUCTION(register_reduction_plus_float,
                  safe_reduce_plus_float, safe_reduce_plus_float_point,
                  reduce_plus_float, reduce_plus_float_point,
                  PlusOpFloat, float, int, ADD, ADD, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double,
                  safe_reduce_plus_double, safe_reduce_plus_double_point,
                  reduce_plus_double, reduce_plus_double_point,
                  PlusOpDouble, double, size_t, ADD, ADD, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32,
                  safe_reduce_plus_int32, safe_reduce_plus_int32_point,
                  reduce_plus_int32, reduce_plus_int32_point,
                  PlusOpInt, int, int, ADD, ADD, 0)
DECLARE_REDUCTION(register_reduction_plus_int64,
                  safe_reduce_plus_int64, safe_reduce_plus_int64_point,
                  reduce_plus_int64, reduce_plus_int64_point,
                  PlusOpLongLong, long long int, long long int, ADD, ADD, 0)
DECLARE_REDUCTION(register_reduction_plus_uint32,
                  safe_reduce_plus_uint32, safe_reduce_plus_uint32_point,
                  reduce_plus_uint32, reduce_plus_uint32_point,
                  PlusOpUInt, unsigned, unsigned, ADD, ADD, 0U)
DECLARE_REDUCTION(register_reduction_plus_uint64,
                  safe_reduce_plus_uint64, safe_reduce_plus_uint64_point,
                  reduce_plus_uint64, reduce_plus_uint64_point,
                  PlusOpULongLong, unsigned long long, unsigned long long,
                  ADD, ADD, 0ULL)

DECLARE_REDUCTION(register_reduction_minus_float,
                  safe_reduce_minus_float, safe_reduce_minus_float_point,
                  reduce_minus_float, reduce_minus_float_point,
                  MinusOpFloat, float, int, ADD, SUB, 0.0f)
DECLARE_REDUCTION(register_reduction_minus_double,
                  safe_reduce_minus_double, safe_reduce_minus_double_point,
                  reduce_minus_double, reduce_minus_double_point,
                  MinusOpDouble, double, size_t, ADD, SUB, 0.0)
DECLARE_REDUCTION(register_reduction_minus_int32,
                  safe_reduce_minus_int32, safe_reduce_minus_int32_point,
                  reduce_minus_int32, reduce_minus_int32_point,
                  MinusOpInt, int, int, ADD, SUB, 0)
DECLARE_REDUCTION(register_reduction_minus_int64,
                  safe_reduce_minus_int64, safe_reduce_minus_int64_point,
                  reduce_minus_int64, reduce_minus_int64_point,
                  MinusOpLongLong, long long int, long long int, ADD, SUB, 0)
DECLARE_REDUCTION(register_reduction_minus_uint32,
                  safe_reduce_minus_uint32, safe_reduce_minus_uint32_point,
                  reduce_minus_uint32, reduce_minus_uint32_point,
                  MinusOpUInt, unsigned, unsigned, ADD, SUB, 0U)
DECLARE_REDUCTION(register_reduction_minus_uint64,
                  safe_reduce_minus_uint64, safe_reduce_minus_uint64_point,
                  reduce_minus_uint64, reduce_minus_uint64_point,
                  MinusOpULongLong, unsigned long long, unsigned long long,
                  ADD, SUB, 0ULL)

DECLARE_REDUCTION(register_reduction_times_float,
                  safe_reduce_times_float, safe_reduce_times_float_point,
                  reduce_times_float, reduce_times_float_point,
                  TImesOPFloat, float, int, MUL, MUL, 1.0f)
DECLARE_REDUCTION(register_reduction_times_double,
                  safe_reduce_times_double, safe_reduce_times_double_point,
                  reduce_times_double, reduce_times_double_point,
                  TimesOpDouble, double, size_t, MUL, MUL, 1.0)
DECLARE_REDUCTION(register_reduction_times_int32,
                  safe_reduce_times_int32, safe_reduce_times_int32_point,
                  reduce_times_int32, reduce_times_int32_point,
                  TimesOpInt, int, int, MUL, MUL, 1)
DECLARE_REDUCTION(register_reduction_times_int64,
                  safe_reduce_times_int64, safe_reduce_times_int64_point,
                  reduce_times_int64, reduce_times_int64_point,
                  TimesOpLongLong, long long int, long long int, MUL, MUL, 1)
DECLARE_REDUCTION(register_reduction_times_uint32,
                  safe_reduce_times_uint32, safe_reduce_times_uint32_point,
                  reduce_times_uint32, reduce_times_uint32_point,
                  TimesOpUInt, unsigned, unsigned, MUL, MUL, 1U)
DECLARE_REDUCTION(register_reduction_times_uint64,
                  safe_reduce_times_uint64, safe_reduce_times_uint64_point,
                  reduce_times_uint64, reduce_times_uint64_point,
                  TimesOpULongLong, unsigned long long, unsigned long long,
                  MUL, MUL, 1ULL)

DECLARE_REDUCTION(register_reduction_divide_float,
                  safe_reduce_divide_float, safe_reduce_divide_float_point,
                  reduce_divide_float, reduce_divide_float_point,
                  DivideOPFloat, float, int, DIV, MUL, 1.0f)
DECLARE_REDUCTION(register_reduction_divide_double,
                  safe_reduce_divide_double, safe_reduce_divide_double_point,
                  reduce_divide_double, reduce_divide_double_point,
                  DivideOpDouble, double, size_t, DIV, MUL, 1.0)
DECLARE_REDUCTION(register_reduction_divide_int32,
                  safe_reduce_divide_int32, safe_reduce_divide_int32_point,
                  reduce_divide_int32, reduce_divide_int32_point,
                  DivideOpInt, int, int, DIV, MUL, 1)
DECLARE_REDUCTION(register_reduction_divide_int64,
                  safe_reduce_divide_int64, safe_reduce_divide_int64_point,
                  reduce_divide_int64, reduce_divide_int64_point,
                  DivideOpLongLong, long long int, long long int, DIV, MUL, 1)
DECLARE_REDUCTION(register_reduction_divide_uint32,
                  safe_reduce_divide_uint32, safe_reduce_divide_uint32_point,
                  reduce_divide_uint32, reduce_divide_uint32_point,
                  DivideOpUInt, unsigned, unsigned, DIV, MUL, 1U)
DECLARE_REDUCTION(register_reduction_divide_uint64,
                  safe_reduce_divide_uint64, safe_reduce_divide_uint64_point,
                  reduce_divide_uint64, reduce_divide_uint64_point,
                  DivideOpULongLong, unsigned long long, unsigned long long,
                  DIV, MUL, 1ULL)

DECLARE_REDUCTION(register_reduction_max_float,
                  safe_reduce_max_float, safe_reduce_max_float_point,
                  reduce_max_float, reduce_max_float_point,
                  MaxOPFloat, float, int, std::max, std::max, -std::numeric_limits<float>::infinity())
DECLARE_REDUCTION(register_reduction_max_double,
                  safe_reduce_max_double, safe_reduce_max_double_point,
                  reduce_max_double, reduce_max_double_point,
                  MaxOpDouble, double, size_t, std::max, std::max, -std::numeric_limits<double>::infinity())
DECLARE_REDUCTION(register_reduction_max_int32,
                  safe_reduce_max_int32, safe_reduce_max_int32_point,
                  reduce_max_int32, reduce_max_int32_point,
                  MaxOpInt, int, int, std::max, std::max, INT_MIN)
DECLARE_REDUCTION(register_reduction_max_int64,
                  safe_reduce_max_int64, safe_reduce_max_int64_point,
                  reduce_max_int64, reduce_max_int64_point,
                  MaxOpLongLong, long long int, long long int, std::max, std::max, LLONG_MIN)
DECLARE_REDUCTION(register_reduction_max_uint32,
                  safe_reduce_max_uint32, safe_reduce_max_uint32_point,
                  reduce_max_uint32, reduce_max_uint32_point,
                  MaxOpUInt, unsigned, unsigned, std::max, std::max,
                  std::numeric_limits<unsigned>::min())
DECLARE_REDUCTION(register_reduction_max_uint64,
                  safe_reduce_max_uint64, safe_reduce_max_uint64_point,
                  reduce_max_uint64, reduce_max_uint64_point,
                  MaxOpULongLong, unsigned long long, unsigned long long,
                  std::max, std::max, std::numeric_limits<unsigned long long>::min())

DECLARE_REDUCTION(register_reduction_min_float,
                  safe_reduce_min_float, safe_reduce_min_float_point,
                  reduce_min_float, reduce_min_float_point,
                  MinOPFloat, float, int, std::min, std::min, std::numeric_limits<float>::infinity())
DECLARE_REDUCTION(register_reduction_min_double,
                  safe_reduce_min_double, safe_reduce_min_double_point,
                  reduce_min_double, reduce_min_double_point,
                  MinOpDouble, double, size_t, std::min, std::min, std::numeric_limits<double>::infinity())
DECLARE_REDUCTION(register_reduction_min_int32,
                  safe_reduce_min_int32, safe_reduce_min_int32_point,
                  reduce_min_int32, reduce_min_int32_point,
                  MinOpInt, int, int, std::min, std::min, INT_MAX)
DECLARE_REDUCTION(register_reduction_min_int64,
                  safe_reduce_min_int64, safe_reduce_min_int64_point,
                  reduce_min_int64, reduce_min_int64_point,
                  MinOpLongLong, long long int, long long int, std::min, std::min, LLONG_MAX)
DECLARE_REDUCTION(register_reduction_min_uint32,
                  safe_reduce_min_uint32, safe_reduce_min_uint32_point,
                  reduce_min_uint32, reduce_min_uint32_point,
                  MinOpUInt, unsigned, unsigned,
                  std::min, std::min, std::numeric_limits<unsigned>::max())
DECLARE_REDUCTION(register_reduction_min_uint64,
                  safe_reduce_min_uint64, safe_reduce_min_uint64_point,
                  reduce_min_uint64, reduce_min_uint64_point,
                  MinOpULongLong, unsigned long long, unsigned long long,
                  std::min, std::min, std::numeric_limits<unsigned long long>::max())
#undef DECLARE_REDUCTION


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
    void REG(legion_reduction_op_id_t redop_id, unsigned array_size)         \
    {                                                                        \
      ArrayReductionOp<CLASS> *op = new ArrayReductionOp<CLASS>(array_size); \
      Runtime::register_reduction_op(redop_id, op);                          \
    }                                                                        \
  }

DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_float, PlusOpFloat)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_double, PlusOpDouble)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_int32, PlusOpInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_int64, PlusOpLongLong)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_uint32, PlusOpUInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_plus_uint64, PlusOpULongLong)

DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_float, MinusOpFloat)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_double, MinusOpDouble)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_int32, MinusOpInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_int64, MinusOpLongLong)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_uint32, MinusOpUInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_minus_uint64, MinusOpULongLong)

DECLARE_ARRAY_REDUCTION(register_array_reduction_times_float, TImesOPFloat)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_double, TimesOpDouble)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_int32, TimesOpInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_int64, TimesOpLongLong)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_uint32, TimesOpUInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_times_uint64, TimesOpULongLong)

DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_float, DivideOPFloat)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_double, DivideOpDouble)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_int32, DivideOpInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_int64, DivideOpLongLong)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_uint32, DivideOpUInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_divide_uint64, DivideOpULongLong)

DECLARE_ARRAY_REDUCTION(register_array_reduction_max_float, MaxOPFloat)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_double, MaxOpDouble)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_int32, MaxOpInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_int64, MaxOpLongLong)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_uint32, MaxOpUInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_max_uint64, MaxOpULongLong)

DECLARE_ARRAY_REDUCTION(register_array_reduction_min_float, MinOPFloat)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_double, MinOpDouble)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_int32, MinOpInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_int64, MinOpLongLong)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_uint32, MinOpUInt)
DECLARE_ARRAY_REDUCTION(register_array_reduction_min_uint64, MinOpULongLong)

#undef DECLARE_ARRAY_REDUCTION
