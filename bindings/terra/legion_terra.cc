/* Copyright 2017 Stanford University
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

#include "legion.h"
#include "legion_terra.h"
#include "legion_terra_tasks.h"
#include "legion_c.h"
#include "legion_c_util.h"

using namespace std;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor::AccessorType;

typedef CObjectWrapper::AccessorGeneric AccessorGeneric;

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
  void REG(legion_reduction_op_id_t redop)                              \
  {                                                                     \
    HighLevelRuntime::register_reduction_op<CLASS>(redop);              \
  }                                                                     \
  void SRED(legion_accessor_generic_t accessor_,                        \
           legion_ptr_t ptr_, T value)                                  \
  {                                                                     \
    AccessorGeneric* accessor = CObjectWrapper::unwrap(accessor_);      \
    ptr_t ptr = CObjectWrapper::unwrap(ptr_);                           \
    accessor->typeify<T>().convert<ReductionFold<CLASS> >().reduce(ptr, value); \
  }                                                                     \
  void SRED_DP(legion_accessor_generic_t accessor_,                     \
               legion_domain_point_t dp_, T value)                      \
  {                                                                     \
    AccessorGeneric* accessor = CObjectWrapper::unwrap(accessor_);      \
    DomainPoint dp = CObjectWrapper::unwrap(dp_);                       \
    accessor->typeify<T>()/*.convert<ReductionFold<CLASS> >()*/.reduce<CLASS>(dp, value); \
  }                                                                     \
  void RED(legion_accessor_generic_t accessor_,                         \
           legion_ptr_t ptr_, T value)                                  \
  {                                                                     \
    AccessorGeneric* accessor = CObjectWrapper::unwrap(accessor_);      \
    ptr_t ptr = CObjectWrapper::unwrap(ptr_);                           \
    accessor->typeify<T>().reduce<CLASS>(ptr, value);                   \
  }                                                                     \
  void RED_DP(legion_accessor_generic_t accessor_,                      \
              legion_domain_point_t dp_, T value)                       \
  {                                                                     \
    AccessorGeneric* accessor = CObjectWrapper::unwrap(accessor_);      \
    DomainPoint dp = CObjectWrapper::unwrap(dp_);                       \
    accessor->typeify<T>().reduce<CLASS>(dp, value);                    \
  }                                                                     \
  }                                                                     \

DECLARE_REDUCTION(register_reduction_plus_float,
                  safe_reduce_plus_float, safe_reduce_plus_float_domain_point,
                  reduce_plus_float, reduce_plus_float_domain_point,
                  PlusOpFloat, float, int, ADD, ADD, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double,
                  safe_reduce_plus_double, safe_reduce_plus_double_domain_point,
                  reduce_plus_double, reduce_plus_double_domain_point,
                  PlusOpDouble, double, size_t, ADD, ADD, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32,
                  safe_reduce_plus_int32, safe_reduce_plus_int32_domain_point,
                  reduce_plus_int32, reduce_plus_int32_domain_point,
                  PlusOpInt, int, int, ADD, ADD, 0)
DECLARE_REDUCTION(register_reduction_plus_int64,
                  safe_reduce_plus_int64, safe_reduce_plus_int64_domain_point,
                  reduce_plus_int64, reduce_plus_int64_domain_point,
                  PlusOpLongLong, long long int, long long int, ADD, ADD, 0)

DECLARE_REDUCTION(register_reduction_minus_float,
                  safe_reduce_minus_float, safe_reduce_minus_float_domain_point,
                  reduce_minus_float, reduce_minus_float_domain_point,
                  MinusOpFloat, float, int, ADD, SUB, 0.0f)
DECLARE_REDUCTION(register_reduction_minus_double,
                  safe_reduce_minus_double, safe_reduce_minus_double_domain_point,
                  reduce_minus_double, reduce_minus_double_domain_point,
                  MinusOpDouble, double, size_t, ADD, SUB, 0.0)
DECLARE_REDUCTION(register_reduction_minus_int32,
                  safe_reduce_minus_int32, safe_reduce_minus_int32_domain_point,
                  reduce_minus_int32, reduce_minus_int32_domain_point,
                  MinusOpInt, int, int, ADD, SUB, 0)
DECLARE_REDUCTION(register_reduction_minus_int64,
                  safe_reduce_minus_int64, safe_reduce_minus_int64_domain_point,
                  reduce_minus_int64, reduce_minus_int64_domain_point,
                  MinusOpLongLong, long long int, long long int, ADD, SUB, 0)

DECLARE_REDUCTION(register_reduction_times_float,
                  safe_reduce_times_float, safe_reduce_times_float_domain_point,
                  reduce_times_float, reduce_times_float_domain_point,
                  TImesOPFloat, float, int, MUL, MUL, 1.0f)
DECLARE_REDUCTION(register_reduction_times_double,
                  safe_reduce_times_double, safe_reduce_times_double_domain_point,
                  reduce_times_double, reduce_times_double_domain_point,
                  TimesOpDouble, double, size_t, MUL, MUL, 1.0)
DECLARE_REDUCTION(register_reduction_times_int32,
                  safe_reduce_times_int32, safe_reduce_times_int32_domain_point,
                  reduce_times_int32, reduce_times_int32_domain_point,
                  TimesOpInt, int, int, MUL, MUL, 1)
DECLARE_REDUCTION(register_reduction_times_int64,
                  safe_reduce_times_int64, safe_reduce_times_int64_domain_point,
                  reduce_times_int64, reduce_times_int64_domain_point,
                  TimesOpLongLong, long long int, long long int, MUL, MUL, 1)

DECLARE_REDUCTION(register_reduction_divide_float,
                  safe_reduce_divide_float, safe_reduce_divide_float_domain_point,
                  reduce_divide_float, reduce_divide_float_domain_point,
                  DivideOPFloat, float, int, DIV, MUL, 1.0f)
DECLARE_REDUCTION(register_reduction_divide_double,
                  safe_reduce_divide_double, safe_reduce_divide_double_domain_point,
                  reduce_divide_double, reduce_divide_double_domain_point,
                  DivideOpDouble, double, size_t, DIV, MUL, 1.0)
DECLARE_REDUCTION(register_reduction_divide_int32,
                  safe_reduce_divide_int32, safe_reduce_divide_int32_domain_point,
                  reduce_divide_int32, reduce_divide_int32_domain_point,
                  DivideOpInt, int, int, DIV, MUL, 1)
DECLARE_REDUCTION(register_reduction_divide_int64,
                  safe_reduce_divide_int64, safe_reduce_divide_int64_domain_point,
                  reduce_divide_int64, reduce_divide_int64_domain_point,
                  DivideOpLongLong, long long int, long long int, DIV, MUL, 1)

DECLARE_REDUCTION(register_reduction_max_float,
                  safe_reduce_max_float, safe_reduce_max_float_domain_point,
                  reduce_max_float, reduce_max_float_domain_point,
                  MaxOPFloat, float, int, max, max, -std::numeric_limits<float>::infinity())
DECLARE_REDUCTION(register_reduction_max_double,
                  safe_reduce_max_double, safe_reduce_max_double_domain_point,
                  reduce_max_double, reduce_max_double_domain_point,
                  MaxOpDouble, double, size_t, max, max, -std::numeric_limits<double>::infinity())
DECLARE_REDUCTION(register_reduction_max_int32,
                  safe_reduce_max_int32, safe_reduce_max_int32_domain_point,
                  reduce_max_int32, reduce_max_int32_domain_point,
                  MaxOpInt, int, int, max, max, INT_MIN)
DECLARE_REDUCTION(register_reduction_max_int64,
                  safe_reduce_max_int64, safe_reduce_max_int64_domain_point,
                  reduce_max_int64, reduce_max_int64_domain_point,
                  MaxOpLongLong, long long int, long long int, max, max, LLONG_MIN)

DECLARE_REDUCTION(register_reduction_min_float,
                  safe_reduce_min_float, safe_reduce_min_float_domain_point,
                  reduce_min_float, reduce_min_float_domain_point,
                  MinOPFloat, float, int, min, min, std::numeric_limits<float>::infinity())
DECLARE_REDUCTION(register_reduction_min_double,
                  safe_reduce_min_double, safe_reduce_min_double_domain_point,
                  reduce_min_double, reduce_min_double_domain_point,
                  MinOpDouble, double, size_t, min, min, std::numeric_limits<double>::infinity())
DECLARE_REDUCTION(register_reduction_min_int32,
                  safe_reduce_min_int32, safe_reduce_min_int32_domain_point,
                  reduce_min_int32, reduce_min_int32_domain_point,
                  MinOpInt, int, int, min, min, INT_MAX)
DECLARE_REDUCTION(register_reduction_min_int64,
                  safe_reduce_min_int64, safe_reduce_min_int64_domain_point,
                  reduce_min_int64, reduce_min_int64_domain_point,
                  MinOpLongLong, long long int, long long int, min, min, LLONG_MAX)

