/* Copyright 2015 Stanford University
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
#include <vector>

#include "legion.h"
#include "legion_terra.h"
#include "lua_mapper_wrapper.h"
#include "legion_c_util.h"
#include "legion_terra_util.h"

using namespace std;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor::AccessorType;

typedef CObjectWrapper::AccessorGeneric AccessorGeneric;

extern "C"
{
#include "lua.h"
#include "terra.h"
#include "legion_c.h"
}

#ifdef PROF_BINDING
static LegionRuntime::Logger::Category log("legion_terra");
#endif

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
    U *target = (U *)&(lhs);                                            \
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
    U *target = (U *)&rhs1;                                             \
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

DECLARE_REDUCTION(register_reduction_max_float,
                  safe_reduce_max_float, safe_reduce_max_float_domain_point,
                  reduce_max_float, reduce_max_float_domain_point,
                  MaxOPFloat, float, int, max, max, -FLT_MAX)
DECLARE_REDUCTION(register_reduction_max_double,
                  safe_reduce_max_double, safe_reduce_max_double_domain_point,
                  reduce_max_double, reduce_max_double_domain_point,
                  MaxOpDouble, double, size_t, max, max, -DBL_MAX)
DECLARE_REDUCTION(register_reduction_max_int32,
                  safe_reduce_max_int32, safe_reduce_max_int32_domain_point,
                  reduce_max_int32, reduce_max_int32_domain_point,
                  MaxOpInt, int, int, max, max, INT_MIN)

DECLARE_REDUCTION(register_reduction_min_float,
                  safe_reduce_min_float, safe_reduce_min_float_domain_point,
                  reduce_min_float, reduce_min_float_domain_point,
                  MinOPFloat, float, int, min, min, FLT_MAX)
DECLARE_REDUCTION(register_reduction_min_double,
                  safe_reduce_min_double, safe_reduce_min_double_domain_point,
                  reduce_min_double, reduce_min_double_domain_point,
                  MinOpDouble, double, size_t, min, min, DBL_MAX)
DECLARE_REDUCTION(register_reduction_min_int32,
                  safe_reduce_min_int32, safe_reduce_min_int32_domain_point,
                  reduce_min_int32, reduce_min_int32_domain_point,
                  MinOpInt, int, int, min, min, INT_MAX)

extern "C"
{

lua_State* prepare_interpreter(const string& script_file)
{
  lua_State* L = luaL_newstate();
  luaL_openlibs(L);
  terra_init(L);
  lua_pushinteger(L, 1);
  lua_setglobal(L, "initialized");
  {
    int err = terra_dofile(L, script_file.c_str());
    if (err != 0)
    {
      fprintf(stderr, "error loading task file : %s\n",
          lua_tostring(L, -1));
      exit(-1);
    }
  }
  return L;
}

static string qualified_callback_name;

void set_lua_registration_callback_name(char* qualified_callback_name_)
{
  qualified_callback_name = qualified_callback_name_;
}

legion_mapper_t create_mapper(const char* qualified_mapper_name,
                              legion_machine_t machine_,
                              legion_runtime_t runtime_,
                              legion_processor_t proc_)
{
  Machine *machine = CObjectWrapper::unwrap(machine_);
  HighLevelRuntime *runtime = CObjectWrapper::unwrap(runtime_);
  Processor proc = CObjectWrapper::unwrap(proc_);

  Mapper *mapper = new LuaMapperWrapper(qualified_mapper_name,
                                        *machine, runtime, proc);
  return CObjectWrapper::wrap(mapper);
}

void lua_registration_callback_wrapper(legion_machine_t machine,
                                       legion_runtime_t runtime,
                                       const legion_processor_t *local_procs,
                                       unsigned num_local_procs)
{
  unsigned n = qualified_callback_name.find_last_of("/");
  string script_file = qualified_callback_name.substr(0, n);
  string callback_name = qualified_callback_name.substr(n + 1);

  lua_State* L = prepare_interpreter(script_file);

  lua_getglobal(L, "lua_registration_callback_wrapper_in_lua");
  lua_pushstring(L, script_file.c_str());
  lua_pushstring(L, callback_name.c_str());
  lua_push_opaque_object<CObjectWrapper>(L, machine);
  lua_push_opaque_object<CObjectWrapper>(L, runtime);
  lua_pushlightuserdata(L, (void*)local_procs);
  lua_pushinteger(L, num_local_procs);

  if (lua_pcall(L, 6, 0, 0) != 0)
  {
    fprintf(stderr,
        "error running lua_registration_callback_wrapper_in_lua : %s\n",
        lua_tostring(L, -1));
    exit(-1);
  }

  lua_close(L);
}

void lua_task_wrapper_void(legion_task_t _task,
                           const legion_physical_region_t* _regions,
                           unsigned _num_regions,
                           legion_context_t _ctx,
                           legion_runtime_t _runtime)
{
#ifdef PROF_BINDING
  double ts_start = LegionRuntime::TimeStamp::get_current_time_in_micros();
#endif

  Task* task = CObjectWrapper::unwrap(_task);
  string qualified_task_name(task->variants->name);
  unsigned n = qualified_task_name.find_last_of("/");
  string script_file = qualified_task_name.substr(0, n);
  string task_name = qualified_task_name.substr(n + 1);

  lua_State* L = prepare_interpreter(script_file);

  lua_getglobal(L, "lua_void_task_wrapper_in_lua");
  lua_pushstring(L, script_file.c_str());
  lua_pushstring(L, task_name.c_str());
  lua_push_opaque_object<CObjectWrapper>(L, _task);
  lua_push_opaque_object_array(L, _regions, _num_regions);
  lua_push_opaque_object<CObjectWrapper>(L, _ctx);
  lua_push_opaque_object<CObjectWrapper>(L, _runtime);

#ifdef PROF_BINDING
  double ts_mid = LegionRuntime::TimeStamp::get_current_time_in_micros();
#endif

  if (lua_pcall(L, 6, 0, 0) != 0)
  {
    fprintf(stderr,
        "error running lua_void_task_wrapper_in_lua : %s\n",
        lua_tostring(L, -1));
    exit(-1);
  }

  lua_close(L);

#ifdef PROF_BINDING
  double ts_end = LegionRuntime::TimeStamp::get_current_time_in_micros();
  log.info(
      "Task id: %llu, Task name: %s, Init time: %.3f ms, Task time: %.3f ms",
      task->get_unique_task_id(),
      task_name.c_str(),
      (ts_mid - ts_start) / 1e3,
      (ts_end - ts_mid) / 1e3);
#endif
}

legion_task_result_t lua_task_wrapper(legion_task_t _task,
                                      const legion_physical_region_t* _regions,
                                      unsigned _num_regions,
                                      legion_context_t _ctx,
                                      legion_runtime_t _runtime)
{
#ifdef PROF_BINDING
  double ts_start = LegionRuntime::TimeStamp::get_current_time_in_micros();
#endif

  Task* task = CObjectWrapper::unwrap(_task);
  string qualified_task_name(task->variants->name);
  unsigned n1 = qualified_task_name.find_last_of("/");
  unsigned n2 = qualified_task_name.find_last_of(":");
  string script_file = qualified_task_name.substr(0, n1);
  string task_name = qualified_task_name.substr(n1 + 1, n2 - (n1 + 1));
  string return_type_name = qualified_task_name.substr(n2 + 1);

  lua_State* L = prepare_interpreter(script_file);

  lua_getglobal(L, "lua_task_wrapper_in_lua");
  lua_pushstring(L, script_file.c_str());
  lua_pushstring(L, return_type_name.c_str());
  lua_pushstring(L, task_name.c_str());
  lua_push_opaque_object<CObjectWrapper>(L, _task);
  lua_push_opaque_object_array(L, _regions, _num_regions);
  lua_push_opaque_object<CObjectWrapper>(L, _ctx);
  lua_push_opaque_object<CObjectWrapper>(L, _runtime);

#ifdef PROF_BINDING
  double ts_mid = LegionRuntime::TimeStamp::get_current_time_in_micros();
#endif

  if (lua_pcall(L, 7, 1, 0) != 0)
    fprintf(stderr, "error running lua_task_wrapper_in_lua : %s\n",
        lua_tostring(L, -1));

  // the return value from the lua_task_wrapper_in_lua is of cdata type.
  // handles similarly as in tcompiler.cpp:2471.
  legion_task_result_t* result_ptr =
    (legion_task_result_t*)(*(void**)lua_topointer(L, -1));
  lua_pop(L, 1);

  legion_task_result_t result =
    legion_task_result_create(result_ptr->value, result_ptr->value_size);
  free(result_ptr->value);
  free(result_ptr);
  lua_close(L);

#ifdef PROF_BINDING
  double ts_end = LegionRuntime::TimeStamp::get_current_time_in_micros();
  log.info(
      "Task id: %llu, Task name: %s, Init time: %.3f ms, Task time: %.3f ms",
      task->get_unique_task_id(),
      task_name.c_str(),
      (ts_mid - ts_start) / 1e3,
      (ts_end - ts_mid) / 1e3);
#endif

  return result;
}

void
vector_legion_domain_split_push_back(vector_legion_domain_split_t slices_,
                                     legion_domain_split_t slice_)
{
  vector<Mapper::DomainSplit> *slices = ObjectWrapper::unwrap(slices_);
  Mapper::DomainSplit slice = CObjectWrapper::unwrap(slice_);
  slices->push_back(slice);
}

unsigned
vector_legion_domain_split_size(vector_legion_domain_split_t slices_)
{
  vector<Mapper::DomainSplit> *slices = ObjectWrapper::unwrap(slices_);
  return slices->size();
}

legion_domain_split_t
vector_legion_domain_split_get(vector_legion_domain_split_t slices_,
                               unsigned idx)
{
  vector<Mapper::DomainSplit> *slices = ObjectWrapper::unwrap(slices_);
  return CObjectWrapper::wrap((*slices)[idx]);
}

}
