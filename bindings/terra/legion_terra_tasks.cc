/* Copyright 2016 Stanford University
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
#include "legion_c.h"
#include "legion_c_util.h"

#include "legion_terra.h"
#include "legion_terra_tasks.h"
#include "legion_terra_util.h"
#include "lua_mapper_wrapper.h"

// These complain if not explicitly included in C mode.
extern "C"
{
#include "lua.h"
#include "terra.h"
}

using namespace std;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor::AccessorType;

typedef CObjectWrapper::AccessorGeneric AccessorGeneric;

#ifdef PROF_BINDING
static LegionRuntime::Logger::Category log("legion_terra");
#endif

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
vector_legion_task_slice_push_back(vector_legion_task_slice_t slices_,
                                     legion_task_slice_t slice_)
{
  vector<Mapper::TaskSlice> *slices = ObjectWrapper::unwrap(slices_);
  Mapper::TaskSlice slice = CObjectWrapper::unwrap(slice_);
  slices->push_back(slice);
}

unsigned
vector_legion_task_slice_size(vector_legion_task_slice_t slices_)
{
  vector<Mapper::TaskSlice> *slices = ObjectWrapper::unwrap(slices_);
  return slices->size();
}

legion_task_slice_t
vector_legion_task_slice_get(vector_legion_task_slice_t slices_,
                               unsigned idx)
{
  vector<Mapper::TaskSlice> *slices = ObjectWrapper::unwrap(slices_);
  return CObjectWrapper::wrap((*slices)[idx]);
}

}
