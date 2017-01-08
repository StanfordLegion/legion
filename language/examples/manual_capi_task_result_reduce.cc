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
#include <cstdlib>

#include "legion.h"
#include "legion_c_util.h"

using namespace std;
using namespace LegionRuntime::HighLevel;

#include "legion_c.h"
#include "manual_capi_task_result_reduce.h"

#define ADD(x, y) ((x) + (y))
#define MUL(x, y) ((x) * (y))

// Pre-defined reduction operators
#define DECLARE_GLOBAL_REDUCTION(REG, CLASS, T, T_N, U, APPLY_OP, FOLD_OP, N, ID) \
  class CLASS {                                                         \
  public:                                                               \
  typedef TaskResult LHS;                                               \
  typedef TaskResult RHS;                                               \
  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);       \
  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);      \
  static const T_N identity_buffer;                                     \
  static const TaskResult identity;                                     \
  };                                                                    \
                                                                        \
  const T_N CLASS::identity_buffer = { { ID } };                        \
  const TaskResult CLASS::identity((void *)&CLASS::identity_buffer,     \
                                   sizeof(CLASS::identity_buffer));     \
                                                                        \
  template <>                                                           \
  void CLASS::apply<true>(LHS &lhs_, RHS rhs_)                          \
  {                                                                     \
    assert(lhs_.value_size == sizeof(T_N));                             \
    assert(rhs_.value_size == sizeof(T_N));                             \
    T_N &lhs = *(T_N *)(lhs_.value);                                    \
    T_N &rhs = *(T_N *)(rhs_.value);                                     \
    for (int i = 0; i < N; ++i) {                                       \
      lhs.value[i] = APPLY_OP(lhs.value[i], rhs.value[i]);              \
    }                                                                   \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::apply<false>(LHS &lhs_, RHS rhs_)                         \
  {                                                                     \
    assert(lhs_.value_size == sizeof(T_N));                             \
    assert(rhs_.value_size == sizeof(T_N));                             \
    T_N &lhs = *(T_N *)(lhs_.value);                                    \
    T_N &rhs = *(T_N *)(rhs_.value);                                     \
    for (int i = 0; i < N; ++i) {                                       \
      U *target = (U *)&(lhs.value[i]);                                 \
      union { U as_U; T as_T; } oldval, newval;                         \
      do {                                                              \
        oldval.as_U = *target;                                          \
        newval.as_T = APPLY_OP(oldval.as_T, rhs.value[i]);              \
      } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
    }                                                                   \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<true>(RHS &rhs1_, RHS rhs2_)                         \
  {                                                                     \
    assert(rhs1_.value_size == sizeof(T_N));                            \
    assert(rhs2_.value_size == sizeof(T_N));                            \
    T_N &rhs1 = *(T_N *)(rhs1_.value);                                  \
    T_N &rhs2 = *(T_N *)(rhs2_.value);                                   \
    for (int i = 0; i < N; ++i) {                                       \
      rhs1.value[i] = FOLD_OP(rhs1.value[i], rhs2.value[i]);            \
    }                                                                   \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<false>(RHS &rhs1_, RHS rhs2_)                        \
  {                                                                     \
    assert(rhs1_.value_size == sizeof(T_N));                            \
    assert(rhs2_.value_size == sizeof(T_N));                            \
    T_N &rhs1 = *(T_N *)(rhs1_.value);                                  \
    T_N &rhs2 = *(T_N *)(rhs2_.value);                                   \
    for (int i = 0; i < N; ++i) {                                       \
      U *target = (U *)&(rhs1.value[i]);                                \
      union { U as_U; T as_T; } oldval, newval;                         \
      do {                                                              \
        oldval.as_U = *target;                                          \
        newval.as_T = FOLD_OP(oldval.as_T, rhs2.value[i]);              \
      } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
    }                                                                   \
  }                                                                     \
                                                                        \
  extern "C"                                                            \
  {                                                                     \
  void REG(legion_reduction_op_id_t redop)                              \
  {                                                                     \
    HighLevelRuntime::register_reduction_op<CLASS>(redop);              \
  }                                                                     \
  }                                                                     \


DECLARE_GLOBAL_REDUCTION(register_reduction_global_plus_int32,
                  GlobalPlusOpint, int, int_1, int, ADD, ADD, 1, 0)
