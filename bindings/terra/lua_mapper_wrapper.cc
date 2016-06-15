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

#include "legion_terra.h"
#include "lua_mapper_wrapper.h"
#include "legion_terra_util.h"

#include <vector>

extern "C"
{
#include "lua.h"
#include "terra.h"

extern lua_State* prepare_interpreter(const std::string&);
}

using namespace std;
using namespace LegionRuntime::HighLevel;

#ifdef PROF_BINDING
static LegionRuntime::Logger::Category log("legion_terra_mapper");
#endif

enum LuaMapperResult
{
  FALSE_RETURNED = 0,
  TRUE_RETURNED,
  UNDEFINED,
};

LuaMapperWrapper::LuaMapperWrapper(Machine machine, Processor local,
                                   const char *name)
  : DefaultMapper(machine, local, name)
{
}

#if 0
// the resulting stack will have the method on top of its class
// if the return value was 0. Once the method call is returned,
// callers should pop out the remaining class from the stack.
static inline int lua_push_method(lua_State *L,
                                  const char *class_name,
                                  const char *method_name)
{
  lua_getglobal(L, class_name);
  if (!lua_istable(L, -1))
  {
    lua_pop(L, 1);
    return -1;
  }
  lua_pushstring(L, method_name);
  lua_gettable(L, -2);
  if (!lua_isfunction(L, -1))
  {
    lua_pop(L, 2);
    return -2;
  }
  return 0;
}

LuaMapperWrapper::LuaMapperWrapper(const char* qualified_mapper_name_,
                                   Machine machine,
                                   HighLevelRuntime *runtime,
                                   Processor local_proc)
  : DefaultMapper(machine, runtime, local_proc),
    qualified_mapper_name(qualified_mapper_name_),
    mapper_name(),
    L(0)
{
#ifdef PROF_BINDING
  double ts_start = LegionRuntime::TimeStamp::get_current_time_in_micros();
#endif
  unsigned n = qualified_mapper_name.find_last_of("/");
  string script_file = qualified_mapper_name.substr(0, n);
  mapper_name = qualified_mapper_name.substr(n + 1);
  L = prepare_interpreter(script_file);

  CHECK_LUA(lua_push_method(L, "LuaMapperWrapper", "new"));

  legion_machine_t machine_ = CObjectWrapper::wrap(&machine);
  legion_runtime_t runtime_ = CObjectWrapper::wrap(runtime);
  legion_processor_t local_proc_ = CObjectWrapper::wrap(local_proc);

  lua_getglobal(L, "LuaMapperWrapper");
  lua_pushlightuserdata(L, this);
  lua_pushstring(L, mapper_name.c_str());
  lua_push_opaque_object<CObjectWrapper>(L, machine_);
  lua_push_opaque_object<CObjectWrapper>(L, runtime_);
  lua_pushlightuserdata(L, &local_proc_);
  CHECK_LUA(lua_pcall(L, 6, 0, 0));

  // pop off the LuaMapperWrapper class(table)
  lua_pop(L, 1);
#ifdef PROF_BINDING
  double ts_end = LegionRuntime::TimeStamp::get_current_time_in_micros();
  log.info(
      "Proc " IDFMT ", Mapper new: %.3f ms",
      local_proc.id,
      (ts_end - ts_start) / 1e3);
#endif
}

LuaMapperWrapper::~LuaMapperWrapper()
{
#ifdef PROF_BINDING
  double ts_start = LegionRuntime::TimeStamp::get_current_time_in_micros();
#endif
  lua_close(L);
#ifdef PROF_BINDING
  double ts_end = LegionRuntime::TimeStamp::get_current_time_in_micros();
  log.info(
      "Proc " IDFMT ", Mapper delete: %.3f ms",
      local_proc.id,
      (ts_end - ts_start) / 1e3);
#endif
}

void LuaMapperWrapper::slice_domain(const Task *task, const Domain &domain,
                                    vector<DomainSplit> &slices)
{
#ifdef PROF_BINDING
  double ts_start = LegionRuntime::TimeStamp::get_current_time_in_micros();
#endif
  CHECK_LUA(lua_push_method(L, "LuaMapperWrapper", "slice_domain_wrapper"));

  const legion_task_t task_ = CObjectWrapper::wrap_const(task);
  legion_domain_t domain_ = CObjectWrapper::wrap(const_cast<Domain&>(domain));
  vector_legion_domain_split_t slices_ = ObjectWrapper::wrap(&slices);

  // call wrapper method in Lua
  lua_getglobal(L, "LuaMapperWrapper");
  lua_push_opaque_object<CObjectWrapper>(L, task_);
  lua_pushlightuserdata(L, &domain_);
  lua_push_opaque_object<ObjectWrapper, vector_legion_domain_split_t>(L, slices_);
  CHECK_LUA(lua_pcall(L, 4, 1, 0));

  // handle return value
  LuaMapperResult result = (LuaMapperResult)lua_tonumber(L, -1);
  lua_pop(L, 1);

  // pop off the LuaMapperWrapper class(table)
  lua_pop(L, 1);
  if (result == UNDEFINED)
    DefaultMapper::slice_domain(task, domain, slices);

#ifdef PROF_BINDING
  double ts_end = LegionRuntime::TimeStamp::get_current_time_in_micros();
  log.info(
      "Proc " IDFMT ", Slice domain: %.3f ms",
      local_proc.id,
      (ts_end - ts_start) / 1e3);
#endif
}

bool LuaMapperWrapper::map_task(Task *task)
{
#ifdef PROF_BINDING
  double ts_start = LegionRuntime::TimeStamp::get_current_time_in_micros();
#endif
  CHECK_LUA(lua_push_method(L, "LuaMapperWrapper", "map_task_wrapper"));

  legion_task_t task_ = CObjectWrapper::wrap(task);

  // call wrapper method in Lua
  lua_getglobal(L, "LuaMapperWrapper");
  lua_push_opaque_object<CObjectWrapper>(L, task_);
  CHECK_LUA(lua_pcall(L, 2, 1, 0));

  // handle return value
  LuaMapperResult result = (LuaMapperResult)lua_tonumber(L, -1);
  lua_pop(L, 1);

  // pop off the LuaMapperWrapper class(table)
  lua_pop(L, 1);
  if (result == UNDEFINED)
    result = static_cast<LuaMapperResult>(DefaultMapper::map_task(task));

#ifdef PROF_BINDING
  double ts_end = LegionRuntime::TimeStamp::get_current_time_in_micros();
  log.info(
      "Proc " IDFMT ", Map task: %.3f ms",
      local_proc.id,
      (ts_end - ts_start) / 1e3);
#endif
  return static_cast<bool>(result);
}

bool LuaMapperWrapper::map_inline(Inline *inline_operation)
{
#ifdef PROF_BINDING
  double ts_start = LegionRuntime::TimeStamp::get_current_time_in_micros();
#endif
  CHECK_LUA(lua_push_method(L, "LuaMapperWrapper", "map_inline_wrapper"));

  legion_inline_t inline_ = CObjectWrapper::wrap(inline_operation);

  // call wrapper method in Lua
  lua_getglobal(L, "LuaMapperWrapper");
  lua_push_opaque_object<CObjectWrapper>(L, inline_);
  CHECK_LUA(lua_pcall(L, 2, 1, 0));

  // handle return value
  LuaMapperResult result = (LuaMapperResult)lua_tonumber(L, -1);
  lua_pop(L, 1);

  // pop off the LuaMapperWrapper class(table)
  lua_pop(L, 1);

  if (result == UNDEFINED)
    result =
      static_cast<LuaMapperResult>(DefaultMapper::map_inline(inline_operation));

#ifdef PROF_BINDING
  double ts_end = LegionRuntime::TimeStamp::get_current_time_in_micros();
  log.info(
      "Proc " IDFMT ", Map inline: %.3f ms",
      local_proc.id,
      (ts_end - ts_start) / 1e3);
#endif
  return static_cast<bool>(result);
}

void LuaMapperWrapper::notify_mapping_failed(const Mappable *mappable)
{
  CHECK_LUA(lua_push_method(L, "LuaMapperWrapper",
        "notify_mapping_failed_wrapper"));

  legion_mappable_t mappable_ =
    CObjectWrapper::wrap(const_cast<Mappable*>(mappable));

  // call wrapper method in Lua
  lua_getglobal(L, "LuaMapperWrapper");
  lua_push_opaque_object<CObjectWrapper>(L, mappable_);
  CHECK_LUA(lua_pcall(L, 2, 1, 0));

  // handle return value
  LuaMapperResult result = (LuaMapperResult)lua_tonumber(L, -1);
  lua_pop(L, 1);

  // pop off the LuaMapperWrapper class(table)
  lua_pop(L, 1);

  if (result == UNDEFINED)
    DefaultMapper::notify_mapping_failed(mappable);
}

extern "C"
{

void decompose_index_space(legion_domain_t domain_,
                           legion_processor_t *targets_,
                           unsigned targets_size,
                           unsigned splitting_factor,
                           vector_legion_domain_split_t slices_)
{
  Domain domain = CObjectWrapper::unwrap(domain_);
  vector<Processor> targets;
  for (unsigned i = 0; i < targets_size; ++i)
    targets.push_back(CObjectWrapper::unwrap(targets_[i]));

  vector<Mapper::DomainSplit>* slices = ObjectWrapper::unwrap(slices_);
  DefaultMapper::decompose_index_space(domain, targets, splitting_factor,
                                       *slices);
}

}
#endif
