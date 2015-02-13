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

#include <cstdio>
#include <string>
#include <cstdlib>

#include "legion.h"
#include "legion_c_util.h"

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
LegionRuntime::Logger::Category log("legion_terra");
#endif

// Pre-defined reduction operators
#define DECLARE_REDUCTION(REG, SRED, RED, CLASS, T, U, OP1, OP2, ID)    \
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
    lhs OP2 rhs;                                                        \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::apply<false>(LHS &lhs, RHS rhs)                           \
  {                                                                     \
    U *target = (U *)&(lhs);                                            \
    union { U as_U; T as_T; } oldval, newval;                           \
    do {                                                                \
      oldval.as_U = *target;                                            \
      newval.as_T = oldval.as_T OP1 rhs;                                \
    } while(!__sync_bool_compare_and_swap(target, oldval.as_U, newval.as_U)); \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<true>(RHS &rhs1, RHS rhs2)                           \
  {                                                                     \
    rhs1 OP2 rhs2;                                                      \
  }                                                                     \
                                                                        \
  template <>                                                           \
  void CLASS::fold<false>(RHS &rhs1, RHS rhs2)                          \
  {                                                                     \
    U *target = (U *)&rhs1;                                             \
    union { U as_U; T as_T; } oldval, newval;                           \
    do {                                                                \
      oldval.as_U = *target;                                            \
      newval.as_T = oldval.as_T OP1 rhs2;                               \
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
  void RED(legion_accessor_generic_t accessor_,                         \
           legion_ptr_t ptr_, T value)                                  \
  {                                                                     \
    AccessorGeneric* accessor = CObjectWrapper::unwrap(accessor_);      \
    ptr_t ptr = CObjectWrapper::unwrap(ptr_);                           \
    accessor->typeify<T>().reduce<CLASS>(ptr, value);                   \
  }                                                                     \
  }                                                                     \

DECLARE_REDUCTION(register_reduction_plus_float,
                  safe_reduce_plus_float,
                  reduce_plus_float,
                  PlusOpFloat, float, int, +, +=, 0.0f)
DECLARE_REDUCTION(register_reduction_plus_double,
                  safe_reduce_plus_double,
                  reduce_plus_double,
                  PlusOpDouble, double, size_t, +, +=, 0.0)
DECLARE_REDUCTION(register_reduction_plus_int32,
                  safe_reduce_plus_int32,
                  reduce_plus_int32,
                  PlusOpInt, int, int, +, +=, 0)

DECLARE_REDUCTION(register_reduction_minus_float,
                  safe_reduce_minus_float,
                  reduce_minus_float,
                  MinusOpFloat, float, int, -, -=, 0.0f)
DECLARE_REDUCTION(register_reduction_minus_double,
                  safe_reduce_minus_double,
                  reduce_minus_double,
                  MinusOpDouble, double, size_t, -, -=, 0.0)
DECLARE_REDUCTION(register_reduction_minus_int32,
                  safe_reduce_minus_int32,
                  reduce_minus_int32,
                  MinusOpInt, int, int, -, -=, 0)

DECLARE_REDUCTION(register_reduction_times_float,
                  safe_reduce_times_float,
                  reduce_times_float,
                  TImesOPFloat, float, int, *, *=, 0.0f)
DECLARE_REDUCTION(register_reduction_times_double,
                  safe_reduce_times_double,
                  reduce_times_double,
                  TimesOpDouble, double, size_t, *, *=, 0.0)
DECLARE_REDUCTION(register_reduction_times_int32,
                  safe_reduce_times_int32,
                  reduce_times_int32,
                  TimesOpInt, int, int, *, *=, 0)

template<typename T>
void lua_push_opaque_object(lua_State* L, T obj)
{
  void* ptr = CObjectWrapper::unwrap(obj);

  lua_newtable(L);
  lua_pushstring(L, "impl");
  lua_pushlightuserdata(L, ptr);
  lua_settable(L, -3);
}

template<typename T>
void lua_push_opaque_object_array(lua_State* L, T* objs, unsigned num_objs)
{
  lua_newtable(L);
  for(unsigned i = 0; i < num_objs; ++i)
  {
    lua_push_opaque_object(L, objs[i]);
    lua_pushinteger(L, i + 1);
    lua_insert(L, -2);
    lua_settable(L, -3);
  }
}

extern "C"
{

static lua_State* prepare_interpreter(const string& script_file)
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
  lua_pushstring(L, callback_name.c_str());
  lua_push_opaque_object(L, machine);
  lua_push_opaque_object(L, runtime);
  lua_push_opaque_object_array(L, local_procs, num_local_procs);

  if (lua_pcall(L, 4, 0, 0) != 0)
  {
    fprintf(stderr,
        "error running lua_registration_callback_wrapper : %s\n",
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
  lua_push_opaque_object(L, _task);
  lua_push_opaque_object_array(L, _regions, _num_regions);
  lua_push_opaque_object(L, _ctx);
  lua_push_opaque_object(L, _runtime);

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
  lua_push_opaque_object(L, _task);
  lua_push_opaque_object_array(L, _regions, _num_regions);
  lua_push_opaque_object(L, _ctx);
  lua_push_opaque_object(L, _runtime);

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

}
